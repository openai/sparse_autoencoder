import FeatureInfo from "./components/featureInfo"
import React, { useEffect } from "react"
import Welcome from "./welcome"
import { Feature } from "./types"
import FeatureSelect from "./components/featureSelect"
import { useState } from "react"
import { useParams, useNavigate, Link } from "react-router-dom"

import { pathForFeature, DEFAULT_AUTOENCODER, AUTOENCODER_FAMILIES } from "./autoencoder_registry"
      
export default function Feed() {
  const params = useParams();
  const navigate = useNavigate();
  let family = AUTOENCODER_FAMILIES[params.family || DEFAULT_AUTOENCODER.family];
  let feature: Feature = {
    // "layer": parseInt(params.layer),
    "atom": parseInt(params.atom),
    "autoencoder": family.get_ae(params),
  };
  console.log('feature', JSON.stringify(feature, null, 2))

  return (
    <div>
      <div>
        <h2 className="flex flex-row">
          <Link to="/">SAE viewer</Link>
        </h2>
          <FeatureSelect
            init_feature={feature}
            onFeatureChange={(f: Feature) => navigate(pathForFeature(f, {replace: true}))}
            onFeatureSubmit={(f: Feature) => navigate(pathForFeature(f, {replace: true}))}
          />
      </div>
      <br/>
      <div style={{ width: '100%', padding: '', margin: "auto", overflow: "visible" }}>
          <FeatureInfo feature={feature}/>
      </div>
    </div>
  )
}
