from abc import ABC
from typing import Any, Callable

import blobfile as bf


class Explanation(ABC):
    def predict(self, tokens: list[str]) -> list[float]:
        raise NotImplementedError

    def dump(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def load(cls, serialized: Any) -> "Explanation":
        raise NotImplementedError

    def dumpf(self, filename: str):
        d = self.dump()
        assert isinstance(d, bytes)
        with bf.BlobFile(filename, "wb") as f:
            f.write(d)

    @classmethod
    def loadf(cls, filename: str):
        with bf.BlobFile(filename, "rb") as f:
            return cls.load(f.read())


_ANY_TOKEN = "token(*)"
_START_TOKEN = "token(^)"
_SALIENCY_KEY = "<saliency>"


class NtgExplanation(Explanation):
    def __init__(self, trie: dict):
        self.trie = trie

    def todict(self) -> dict:
        return {
            "trie": self.trie,
        }

    @classmethod
    def load(cls, serialized: dict) -> "Explanation":
        assert isinstance(serialized, dict)
        return cls(serialized["trie"])

    def predict(self, tokens: list[str]) -> list[float]:
        predicted_acts = []
        # for each token, traverse the trie beginning from that token and proceeding in reverse order until we match
        # a pattern or are no longer able to traverse.
        for i in range(len(tokens)):
            curr = self.trie
            for j in range(i, -1, -1):
                if tokens[j] not in curr and _ANY_TOKEN not in curr:
                    predicted_acts.append(0)
                    break
                if tokens[j] in curr:
                    curr = curr[tokens[j]]
                else:
                    curr = curr[_ANY_TOKEN]
                if _SALIENCY_KEY in curr:
                    predicted_acts.append(curr[_SALIENCY_KEY])
                    break
                # if we"ve reached the end of the sequence and haven't found a saliency value, append 0.
                elif j == 0:
                    if _START_TOKEN in curr:
                        curr = curr[_START_TOKEN]
                        assert _SALIENCY_KEY in curr
                        predicted_acts.append(curr[_SALIENCY_KEY])
                        break
                    predicted_acts.append(0)
        # We should have appended a value for each token in the sequence.
        assert len(predicted_acts) == len(tokens)
        return predicted_acts

    # TODO make this more efficient
    def predict_many(self, tokens_batch: list[list[str]]) -> list[list[float]]:
        return [self.predict(t) for t in tokens_batch]


def batched(iterable, bs):
    batch = []
    it = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(bs):
                batch.append(next(it))
            yield batch
        except StopIteration:
            if len(batch) > 0:
                yield batch
            return


def apply_batched(fn, iterable, bs):
    for batch in batched(iterable, bs):
        ret = fn(batch)
        assert len(ret) == len(batch)
        yield from ret


def batch_parallelize(algos, fn, batch_size):
    """
    Algorithms are coroutines that yield items to be processed in parallel.
    We concurrently run the algorithm on all items in the batch.
    """
    inputs = []
    for i, algo in enumerate(algos):
        inputs.append((i, next(algo)))
    results = [None] * len(algos)
    while len(inputs) > 0:
        ret = list(apply_batched(fn, [x[1] for x in inputs], batch_size))
        assert len(ret) == len(inputs)
        inds = [x[0] for x in inputs]
        inputs = []
        for i, r in zip(inds, ret):
            try:
                next_input = algos[i].send(r)
                inputs.append((i, next_input))
            except StopIteration as e:
                results[i] = e.value
    return results


def create_n2g_explanation(
    model_fn: Callable, train_set: list[dict], batch_size: int = 16,
    padding_token=4808  # " _" for GPT-2
) -> NtgExplanation:
    truncated = []
    # for each one find the index of the selected activation in the doc. truncate the sequences after this point.
    for doc in train_set:
        # get index of selected activation. for docs stored in 'top', this is the max activation.
        # for docs stored in 'random', it is a random positive activation (we sample activations, not docs
        # to populate 'random', so docs with more positive activations are more likely to be included).
        max_idx = doc["idx"]
        truncated.append(
            {
                "act": doc["act"],
                "acts": doc["acts"][: max_idx + 1],
                "tokens": doc["tokens"][: max_idx + 1],
                "token_ints": doc["token_ints"][: max_idx + 1],
            }
        )

    def get_minimal_subsequence(doc):
        for i in range(len(doc["token_ints"]) - 1, -1, -1):
            atom_acts = yield doc["token_ints"][i:]
            assert (
                len(atom_acts) == len(doc["token_ints"]) - i
            ), f"{len(atom_acts)} != {len(doc['token_ints']) - i}"
            if atom_acts[-1] / doc["act"] >= 0.5:
                return {
                    "tokens": doc["tokens"][i:],
                    "token_ints": doc["token_ints"][i:],
                    "subsequence_act": atom_acts[-1],
                    "orig_act": doc["act"],
                }
        print("Warning: no minimal subsequence found")
        # raise ValueError("No minimal subsequence found")
        return {
            "tokens": doc["tokens"],
            "token_ints": doc["token_ints"],
            "subsequence_act": doc["act"],
            "orig_act": doc["act"],
        }

    minimal_subsequences = batch_parallelize(
        [get_minimal_subsequence(doc) for doc in truncated], model_fn, batch_size
    )

    start_padded = apply_batched(
        model_fn,
        [[padding_token] + doc["token_ints"] for doc in minimal_subsequences],
        batch_size,
    )
    for min_seq, pad_atom_acts in zip(minimal_subsequences, start_padded):
        min_seq["can_pad_start"] = pad_atom_acts[-1] / min_seq["orig_act"] >= 0.5

    # for m in minimal_subsequences:
    #     print("\t" + "".join(m["tokens"]))

    # for each token in a minimal subsequence, replace it with a padding token and compute the saliency value (1 - (orig act / new act))
    for doc in minimal_subsequences:
        all_seqs = []
        for i in range(len(doc["token_ints"])):
            tokens = doc["token_ints"][:i] + [padding_token] + doc["token_ints"][i + 1 :]
            assert len(tokens) == len(doc["token_ints"])
            all_seqs.append(tokens)
        saliency_vals = []
        all_atom_acts = apply_batched(model_fn, all_seqs, batch_size)
        for atom_acts, tokens in zip(all_atom_acts, all_seqs):
            assert len(atom_acts) == len(tokens)
            saliency_vals.append(1 - (atom_acts[-1] / doc["subsequence_act"]))
        doc["saliency_vals"] = saliency_vals

    trie = {}
    for doc in minimal_subsequences:
        curr = trie
        for i, (token, saliency) in enumerate(zip(doc["tokens"][::-1], doc["saliency_vals"][::-1])):
            if saliency < 0.5:
                token = _ANY_TOKEN
            if token not in curr:
                curr[token] = {}
            curr = curr[token]
            if i == len(doc["tokens"]) - 1:
                if not doc["can_pad_start"]:
                    curr[_START_TOKEN] = {}
                    curr = curr[_START_TOKEN]
                curr[_SALIENCY_KEY] = doc["subsequence_act"]

    return NtgExplanation(trie)


if __name__ == "__main__":

    def first_position_fn(tokens: list[list[float]]) -> list[list[float]]:
        return [[1.0] + [0.0] * (len(toks) - 1) for toks in tokens]

    expl = create_n2g_explanation(
        first_position_fn,
        [
            {
                "idx": 0,
                "act": 1.0,
                "acts": [1.0, 0.0, 0.0],
                "tokens": ["a", "b", "c"],
                "token_ints": [0, 1, 2],
            },
            {
                "idx": 0,
                "act": 1.0,
                "acts": [1.0, 0.0, 0.0],
                "tokens": ["b", "c", "d"],
                "token_ints": [1, 2, 3],
            },
        ],
    )
    print(expl.trie)
    assert expl.predict(["a", "b", "c"]) == [1.0, 0.0, 0.0]
    assert expl.predict(["c", "b", "a"]) == [1.0, 0.0, 0.0]

    def c_fn(tokens: list[list[float]]) -> list[list[float]]:
        return [[1.0 if tok == 2 else 0.0 for tok in toks] for toks in tokens]

    expl = create_n2g_explanation(
        c_fn,
        [
            {
                "idx": 2,
                "act": 1.0,
                "acts": [0.0, 0.0, 1.0],
                "tokens": ["a", "b", "c"],
                "token_ints": [0, 1, 2],
            },
            {
                "idx": 1,
                "act": 1.0,
                "acts": [0.0, 1.0, 0.0],
                "tokens": ["b", "c", "d"],
                "token_ints": [1, 2, 3],
            },
        ],
    )

    print(expl.trie)
    assert expl.predict(["b", "c"]) == [0.0, 1.0]
    assert expl.predict(["a", "a", "a"]) == [0.0, 0.0, 0.0]
    assert expl.predict(["c", "b", "c"]) == [1.0, 0.0, 1.0]

    def a_star_c_fn(tokens: list[list[float]]) -> list[list[float]]:
        return [
            [1.0 if (tok == 2 and 0 in toks[:i]) else 0.0 for i, tok in enumerate(toks)]
            for toks in tokens
        ]

    expl = create_n2g_explanation(
        a_star_c_fn,
        [
            {
                "idx": 2,
                "act": 1.0,
                "acts": [0.0, 0.0, 1.0],
                "tokens": ["a", "b", "c"],
                "token_ints": [0, 1, 2],
            },
            {
                "idx": 2,
                "act": 1.0,
                "acts": [0.0, 0.0, 1.0],
                "tokens": ["b", "a", "c"],
                "token_ints": [1, 0, 2],
            },
            {
                "idx": 1,
                "act": 1.0,
                "acts": [0.0, 1.0, 0.0],
                "tokens": ["a", "c", "d"],
                "token_ints": [0, 2, 3],
            },
        ],
    )

    print(expl.trie)
    assert expl.predict(["b", "c"]) == [0.0, 0.0]
    assert expl.predict(["a", "c"]) == [0.0, 1.0]
    assert expl.predict(["a", "e", "c", "a", "c"]) == [0.0, 0.0, 1.0, 0.0, 1.0]
    # NOTE: should be 0, 0, 0, 1 but we're not smart enough to handle this yet
    assert expl.predict(["a", "b", "b", "c"]) == [0.0, 0.0, 0.0, 0.0]

    def zero_fn(tokens: list[list[float]]) -> list[list[float]]:
        return [[0.0 for tok in toks] for toks in tokens]

    expl = create_n2g_explanation(zero_fn, [])

    print(expl.trie)
    assert expl.predict(["b", "c"]) == [0.0, 0.0]
    assert expl.predict(["a", "c"]) == [0.0, 0.0]
    assert expl.predict(["a", "e", "c", "a", "c"]) == [0.0, 0.0, 0.0, 0.0, 0.0]
