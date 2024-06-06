import "./App.css"
import Feed from "./feed"
import React from "react"
import { Routes, Route, HashRouter } from "react-router-dom"
import { AUTOENCODER_FAMILIES } from "./autoencoder_registry"
import Welcome from "./welcome"

function App() {
  return (
    <div style={{ width: '100%', paddingBottom: '20px'}}>
    <HashRouter>
      <Routes>
        <Route path="/" element={<Welcome />} />
        <Route path="/feature/:atom" element={<Feed />} />
        {
          Object.values(AUTOENCODER_FAMILIES).map((family) => {
            let extra = '';
            family.selectors.forEach((selector) => {
              extra += `/${selector.key}/:${selector.key}`;
            })
            return <Route key={family.name} path={`/model/:model/family/:family${extra}/feature/:atom`} element={<Feed />} />
          })
        }
      </Routes>
    </HashRouter>
    </div>
  )
}

export default App
