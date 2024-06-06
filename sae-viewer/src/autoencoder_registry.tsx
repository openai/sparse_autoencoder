import { Autoencoder, Feature } from './types';
import { Route } from "react-router-dom"


export const GPT2_LAYER_FAMILY_32k = {
  subject: 'gpt2-small',
  name: 'v5_32k',
  label: 'n=32768, k=32, all locations',
  selectors: [
    {key: 'layer', label: 'Layer', values: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']},
    {key: 'location', label: 'Location', values: ['resid_post_attn', 'resid_post_mlp']},
  ],
  default_H: (H: {[key: string]: string}) => ({
    layer: H.layer || '8', location: H.location || 'resid_post_mlp'
  }),
  get_ae: (H: {[key: string]: string}) => ({
    subject: 'gpt2-small',
    family: 'v5_32k',
    H: {layer: H.layer, location: H.location},
    path: `v5_32k/layer_${H.layer}/${H.location}`,
    num_features: 32768,
  }),
}

export const GPT2_LAYER_FAMILY_128k = {
  subject: 'gpt2-small',
  name: 'v5_128k',
  label: 'n=131072, k=32, all locations',
  selectors: [
    {key: 'layer', label: 'Layer', values: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']},
    {key: 'location', label: 'Location', values: ['resid_post_attn', 'resid_post_mlp']},
  ],
  default_H: (H: {[key: string]: string}) => ({
    layer: H.layer || '8', location: H.location || 'resid_post_mlp'
  }),
  get_ae: (H: {[key: string]: string}) => ({
    subject: 'gpt2-small',
    family: 'v5_128k',
    H: {layer: H.layer, location: H.location},
    path: `v5_128k/layer_${H.layer}/${H.location}`,
    num_features: 131072,
    no_effects: true,
  }),
}

export const GPT2_NK_SWEEP = {
  subject: 'gpt2-small',
  name: 'v5_l8_postmlp',
  label: 'layer 8, resid post MLP, n/k sweep',
  selectors: [
    {key: 'num_features', label: 'Total features', values: ['2048', '8192', '32768', '131072', '524288', '2097152']},
    {key: 'num_active_features', label: 'Active features', values: ['8', '16', '32', '64', '128', '256', '512']},
  ],
  default_H: (H: {[key: string]: string}) => ({
    num_features: H.num_features || '2097152', num_active_features: H.num_active_features || '16'
  }),
  get_ae: (H: {[key: string]: string}) => ({
    subject: 'gpt2-small',
    family: 'v5_l8_postmlp',
    H: {num_features: H.num_features, num_active_features: H.num_active_features},
    path: `v5_l8_postmlp/n${H.num_features}/k${H.num_active_features}`,
    num_features: H.num_features,
  }),
}

export const GPT4_16m = {
  subject: 'gpt4',
  name: 'v5_latelayer_postmlp',
  label: 'n=16M',
  warning: 'Only 65536 features available.  Activations shown on The Pile (uncopyrighted) instead of our internal training dataset.',
  selectors: [],
  default_H: (H: {[key: string]: string}) => ({}),
  get_ae: (H: {[key: string]: string}) => ({
    subject: 'gpt4',
    family: 'v5_latelayer_postmlp',
    H: {},
    path: `v5_latelayer_postmlp/n16777216/k256`,
    num_features: 65536,
    no_effects: true,
  }),
}

// export const DEFAULT_AUTOENCODER = GPT2_NK_SWEEP.get_ae(
//   GPT2_NK_SWEEP.default_H({})
// );
export const DEFAULT_AUTOENCODER = GPT4_16m.get_ae(
  GPT4_16m.default_H({})
);

export const AUTOENCODER_FAMILIES = Object.fromEntries(
  [
    GPT2_NK_SWEEP,
    GPT2_LAYER_FAMILY_32k,
    GPT2_LAYER_FAMILY_128k,
    GPT4_16m,
  ].map((family) => [family.name, family])
);

export const SUBJECT_MODELS = ['gpt2-small', 'gpt4'];

export function pathForFeature(feature: Feature) {
  let res = `/model/${feature.autoencoder.subject}/family/${feature.autoencoder.family}`;
  // for (const [key, value] of Object.entries(feature.autoencoder.H)) {
  //   res += `/${key}/${value}`;
  // }
  for (const selector of AUTOENCODER_FAMILIES[feature.autoencoder.family].selectors) {
    res += `/${selector.key}/${feature.autoencoder.H[selector.key]}`;
  }
  res += `/feature/${feature.atom}`;
  console.log('res', res)
  return res
}

