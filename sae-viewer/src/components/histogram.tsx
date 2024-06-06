import React from 'react';
import Plot from 'react-plotly.js';

// TODO get from data
const BIN_WIDTH = 0.2;
// # bins_fn = lambda lats: (lats / BIN_WIDTH).ceil().int()
// bin_fn = lambda val: math.ceil(val / BIN_WIDTH)
// bin_id_to_lower_bound = lambda xs: xs * BIN_WIDTH

const HistogramDisplay = ({ data }) => {
    // min_bin = min(hist.keys())
    // max_bin = max(hist.keys())
    // ys = [hist.get(x, 0) for x in np.arange(min_bin, max_bin + 1)]
    // xs = np.arange(min_bin, max_bin + 2)
    // xs = bin_id_to_lower_bound(np.array(xs))
    const min_bin = Math.min(...Object.keys(data).map(Number));
    const max_bin = Math.max(...Object.keys(data).map(Number));
    const ys = Array.from({length: max_bin - min_bin + 1}, (_, i) => data[min_bin + i] || 0);
    let xs = Array.from({length: max_bin - min_bin + 2}, (_, i) => min_bin + i);
    xs = xs.map(x => x * BIN_WIDTH);

    const trace = {
      line: {shape: 'hvh'},
      mode: 'lines',
      type: 'scatter',
      x: xs,
      y: ys,
      fill: 'tozeroy',
    };
    const layout = {
      legend: {
        y: 0.5,
        font: {size: 16},
        traceorder: 'reversed',
      },
      yaxis: {
          type: 'log',
          autorange: true
      },
      margin: { l: 30, r: 0, b: 20, t: 0 },
      autosize: true,
    };

  return (
    <Plot
      style={{width: '100%', height: '100%'}}
      data={[trace]}
      layout={layout}
    />
  )
}

export default HistogramDisplay;
