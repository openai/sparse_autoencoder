import React, { useEffect, useState, useRef } from "react"
import { normalizeSequences, SequenceInfo, Feature, FeatureInfo } from "../types"
import TokenHeatmap from "./tokenHeatmap";
import TokenAblationmap from "./tokenAblationmap";
import Histogram from "./histogram"
import Tooltip from "./tooltip"

import {get_feature_info} from "../interpAPI"

export default ({ feature }: {feature: Feature}) => {
  const [data, setData] = useState(null as FeatureInfo | null)
  const [showingMore, setShowingMore] = useState({})
  const [renderNewlines, setRenderNewlines] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [got_error, setError] = useState(null)
  const currentFeatureRef = useRef(feature);


  useEffect(() => {
    async function fetchData() {
      setIsLoading(true)
      try {
        currentFeatureRef.current = feature; // Update current feature in ref on each effect run
        const result = await get_feature_info(feature)
        if (currentFeatureRef.current !== feature) {
          return;
        }
        normalizeSequences(result.top, result.random)
        result.top.sort((a, b) => b.act - a.act);
        setData(result)
        setIsLoading(false)
        setError(null);
      } catch (e) {
        setError(e);
      }
      try {
        const result = await get_feature_info(feature, true)
        if (currentFeatureRef.current !== feature) {
          return;
        }
        normalizeSequences(result.top, result.random)
        result.top.sort((a, b) => b.act - a.act);
        setData(result)
        setIsLoading(false)
        setError(null);
      } catch (e) {
        setError('Note: ablation effects data not available for this model');
      }
    }
    fetchData()
  }, [feature])

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
        <div>loading top dataset examples</div>
        {
          got_error ? <span style={{color: 'red'}}>Error loading data: {got_error}</span> : null
        }
      </div>
    )
  }
  if (!data) {
    throw new Error('no data.  this should not happen.')
  }

  const all_sequences = []
  all_sequences.push({
    // label: '[0, 1] (Random)',
    label: 'Random positive activations',
    sequences: data.random,
    default_show: 5,
  })
  all_sequences.push({
    // label: '[0.999, 1] (Top quantile, sorted.  50 of 50000)',
    label: 'Top activations',
    sequences: data.top,
    default_show: 5,
  })

  // const activations = data.top_activations;
  return (
    <div>
    {
      got_error ? <span style={{color: '#ee5555'}}>{got_error}</span> : null
    }
    <div style={{flexDirection: 'row', display: 'flex'}}>
      <div style={{width: '500px', height: '200px'}}>
        <Histogram data={data.hist} />
      </div>
      <table style={{marginLeft:'20px'}}>
      <tbody>
        <tr>
        <td><Tooltip content={'Density'} tooltip={'E[a > 0]'}/>
        </td>
        <td>{data.density.toExponential(2)}</td>
        </tr>
        <tr>
        <td><Tooltip content={'Mean'} tooltip={'E[a]'}/>
        </td>
        <td>{data.mean_act ? data.mean_act.toExponential(2) : 'data not available'}</td>
        </tr>
        <tr>
        <td><Tooltip content={'Variance (0 centered)'} tooltip={<>E[a<sup>2</sup>]</>}/></td>
        <td>{data.mean_act_squared ? data.mean_act_squared.toExponential(2): 'data not available'}</td>
        </tr>
        <tr>
        <td><Tooltip content={'Skew (0 centered)'} tooltip={<>E[a<sup>3</sup>]/(E[a<sup>2</sup>])<sup>1.5</sup></>}/></td>
        <td>{data.skew ? data.skew.toExponential(2) : 'data not available'}</td>
        </tr>
        <tr>
        <td><Tooltip content={'Kurtosis (0 centered)'} tooltip={<>E[a<sup>4</sup>]/(E[a<sup>2</sup>])<sup>2</sup></>}/></td>
        <td>{data.kurtosis ? data.kurtosis.toExponential(2) : 'data not available'}</td>
        </tr>
        </tbody>
      </table>
    </div>
      {
        all_sequences.map(({label, sequences, default_show}, idx) => {
          // console.log('sequences', sequences)
          const n_show = showingMore[label] ? sequences.length : default_show;
          return (
          <React.Fragment key={idx}>
          <h3 className="text-md font-bold">
            {label}
            <button className="ml-2 mb-2 mt-2 text-sm text-gray-500"
              onClick={() => setShowingMore({...showingMore, [label]: !showingMore[label]})}>
              {showingMore[label] ? 'show less' : 'show more'}
            </button>
            <button className="ml-2 mb-2 mt-2 text-sm text-gray-500"
              onClick={() => setRenderNewlines(!renderNewlines)}>
              {renderNewlines ? 'collapse newlines' : 'show newlines'}
            </button>
          </h3>
          <table style={{fontSize: '12px'}} className="activations-table" >
            <thead>
            <tr>
            <th>Doc ID</th><th>Token</th><th>Activation</th><th>Activations</th>
            {sequences.length && sequences[0].ablate_loss_diff && <th>Effects</th>}
            </tr>
            </thead>
            <tbody>
            {sequences.slice(0, n_show).map((sequence, i) => (
              <tr key={i}>
              <td className="center">{sequence.doc_id}</td><td className="center">{sequence.idx}</td><td className="center">{sequence.act.toFixed(2)}</td>
              <td className="p-2">
                  <TokenHeatmap info={sequence} renderNewlines={renderNewlines}/>
              </td>
              {
                  sequence.ablate_loss_diff &&
                  <td className="p-2">
                  <TokenAblationmap info={sequence} renderNewlines={renderNewlines}/>
                  </td>
              }
              </tr>
            ))}
            </tbody>
          </table>
          </React.Fragment>
          )
        })
      }
    </div>
  )
}
