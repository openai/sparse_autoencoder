import {Feature, FeatureInfo} from './types';
import {memoizeAsync} from "./utils"

export const load_file_no_cache = async(path: string) => {
  const data = {
    path: path
  }
  const url = new URL("/load_az", window.location.href)
  url.port = '8000';
  return await (
    await fetch(url, {
      method: "POST", // or 'PUT'
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
  ).json()
  
}

export  const load_file_az = async(path: string) => {
  const res = (
    await fetch(path, {
      method: "GET",
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
    })
  )
  if (!res.ok) {
    console.error(`HTTP error: ${res.status} - ${res.statusText}`);
    return;
  }
  return await res.json()
}


// export const load_file = memoizeAsync('load_file', load_file_no_cache)
// export  const load_file = window.location.host.indexOf('localhost:') === -1 ? load_file_az : load_file_no_cache;
export  const load_file = load_file_no_cache;


export async function get_feature_info(feature: Feature, ablated?: boolean): Promise<FeatureInfo> {
  let load_fn = load_file_az;
  let prefix = "https://openaipublic.blob.core.windows.net/sparse-autoencoder/viewer"
  if (window.location.host.indexOf('localhost:') !== -1) {
    load_fn = load_file;
    prefix = "az://openaipublic/sparse-autoencoder/viewer"
    // prefix = az://oaialignment/interp/autoencoder-vis/ae
  }
  
  const ae = feature.autoencoder;
  const result = await load_fn(`${prefix}/${ae.subject}/${ae.path}/atoms/${feature.atom}${ablated ? '-ablated': ''}.json`)
  // console.log('result', result)
  return result
}
