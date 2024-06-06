import React from "react"
import { interpolateColor, Color, getInterpolatedColor, DEFAULT_COLORS, DEFAULT_BOUNDARIES, SequenceInfo } from '../types'
import Tooltip from './tooltip'

type Props = {
  info: SequenceInfo, 
  colors?: Color[], 
  boundaries?: number[]
  renderNewlines?: boolean,
}

function zip_sequence(sequence: SequenceInfo) {
  return sequence.tokens.map((token, idx) => ({
    token,
    highlight: idx === sequence.idx,
    activation: sequence.acts[idx],
    normalized_activation: sequence.normalized_acts ? sequence.normalized_acts[idx] : undefined
  }));
}

export default function TokenHeatmap({ info, colors = DEFAULT_COLORS, boundaries = DEFAULT_BOUNDARIES, renderNewlines }: Props) {
    // <div className="block" style={{width:'100%', whiteSpace: 'pre', overflowX: 'scroll' }}>
  const zipped = zip_sequence(info)
  return (
    <div className="block" style={{width:'100%', whiteSpace: 'pre-wrap'}}>
      {zipped.map(({ token, activation, normalized_activation, highlight }, i) => {
        const color = getInterpolatedColor(colors, boundaries, normalized_activation || activation);
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
          tooltip={<div>Activation: {activation.toFixed(2)}</div>}
          key={i}
          />
      })}
    </div>
  )
}
