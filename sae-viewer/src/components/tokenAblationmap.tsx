import React from "react"
import { interpolateColor, Color, getInterpolatedColor, DEFAULT_COLORS, SequenceInfo } from '../types'
import Tooltip from './tooltip'
import { scaleLinear } from "d3-scale"

type Props = {
  info: SequenceInfo, 
  colors?: Color[], 
  boundaries?: number[],
  renderNewlines?: boolean,
}

export const normalizeToUnitInterval = (arr: number[]) => {
  const max = Math.max(...arr);
  const min = Math.min(...arr);
  const max_abs = Math.max(Math.abs(max), Math.abs(min));
  const rescale = scaleLinear()
    // Even though we're only displaying positive activations, we still need to scale in a way that
    // accounts for the existence of negative activations, since our color scale includes them.
    .domain([-max_abs, max_abs])
    .range([-1, 1])

  return arr.map((x) => rescale(x));
}


export default function TokenAblationmap({ info, colors = DEFAULT_COLORS, renderNewlines }: Props) {
    // <div className="block" style={{width:'100%', whiteSpace: 'pre', overflowX: 'scroll' }}>
  if (!info.ablate_loss_diff) {
    return <> </>;
  }
  const lossDiffsNorm = normalizeToUnitInterval(info.ablate_loss_diff.map((x) => (-x)));
  return (
    <div className="block" style={{width:'100%', whiteSpace: 'pre-wrap'}}>
      {info.tokens.map((token, idx) => {
        const highlight = idx === info.idx;
        const loss_diff = (idx === 0) ? 0: info.ablate_loss_diff[idx-1];
        const kl = (idx === 0) ? 0: info.kl[idx-1];
        const activation = info.acts[idx];
        const top_downvotes = (idx === 0) ? [] : info.top_downvotes_logits[idx-1];
        const top_downvote_tokens = (idx === 0) ? [] : info.top_downvote_tokens_logits[idx-1];
        const top_upvotes = (idx === 0) ? [] : info.top_upvotes_logits[idx-1];
        const top_upvote_tokens = (idx === 0) ? [] : info.top_upvote_tokens_logits[idx-1];
        // const top_downvotes_weighted = (idx === 0) ? [] : info.top_downvotes_weighted[idx-1];
        // const top_downvote_tokens_weighted = (idx === 0) ? [] : info.top_downvote_tokens_weighted[idx-1];
        // const top_upvotes_weighted = (idx === 0) ? [] : info.top_upvotes_weighted[idx-1];
        // const top_upvote_tokens_weighted = (idx === 0) ? [] : info.top_upvote_tokens_weighted[idx-1];
        const top_downvotes_probs = (idx === 0) ? [] : info.top_downvotes_probs[idx-1];
        const top_downvote_tokens_probs = (idx === 0) ? [] : info.top_downvote_tokens_probs[idx-1];
        const top_upvotes_probs = (idx === 0) ? [] : info.top_upvotes_probs[idx-1];
        const top_upvote_tokens_probs = (idx === 0) ? [] : info.top_upvote_tokens_probs[idx-1];
        const color = getInterpolatedColor(colors, [-1, 0, 1], (idx === 0) ? 0 : lossDiffsNorm[idx-1]);
        if (!renderNewlines) {
          token = token.replace(/\n/g, '↵')
        }
        return <Tooltip
          content={
            <span
              style={{
                background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`,
                border: highlight ? '2px solid gray' : 'none',
                borderRadius: '2px',
              }}
                >
              {token}
            </span>
          }
          tooltip={(idx <= info.idx) ? <div>(prediction prior to ablated token)</div> : <div>
            Loss diff:  {loss_diff.toExponential(2)} <br/>
            KL(clean || ablated):  {kl.toExponential(2)} <br/>
            Logit diffs:
              <table>
                <thead>
                <tr>
                {
                  ['', /*' (weighted)',*/ ' (probs)'].map((suffix, i) => {
                    return <React.Fragment key={i}>
                      <th>Upvoted {suffix}</th><th></th>
                      <th>Downvoted {suffix}</th><th></th>
                    </React.Fragment>
                  })
                }
                </tr>
                </thead>
                <tbody>
                  {
                    top_upvotes.map((upvote, j) => {
                      const downvote = top_downvotes[j];
                      return <tr key={j}>
                        <td>{upvote.toExponential(1)}</td>
                        <td style={{whiteSpace: 'pre'}}>{top_upvote_tokens[j]}</td>
                        <td>{downvote.toExponential(1)}</td>
                        <td style={{whiteSpace: 'pre'}}>{top_downvote_tokens[j]}</td>
                        <td>{top_upvotes_probs[j].toExponential(1)}</td>
                        <td style={{whiteSpace: 'pre'}}>{top_upvote_tokens_probs[j]}</td>
                        <td>{top_downvotes_probs[j].toExponential(1)}</td>
                        <td style={{whiteSpace: 'pre'}}>{top_downvote_tokens_probs[j]}</td>
                      </tr>
                    })
                  }
                </tbody>
              </table>
          </div>}
          key={idx}
          />
      })}
    </div>
  )

}
