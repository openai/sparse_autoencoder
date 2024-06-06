import React from "react"
import { useState, FormEvent } from "react"
import { useNavigate } from "react-router-dom"
import { Feature } from "./types"
import FeatureSelect from "./components/featureSelect"
import { pathForFeature, DEFAULT_AUTOENCODER, AUTOENCODER_FAMILIES } from "./autoencoder_registry"

export default function Welcome() {
  const navigate = useNavigate()

  const GPT4_ATOMS_PER_SHARD = 1024;
  const displayFeatures = [
    /**************
    /* well explained + interesting
    ***************/
    {heading: 'GPT-4', heading_type: 'h4', feature: null, label: ''},
    {feature: {atom: 62 * GPT4_ATOMS_PER_SHARD + 53, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
     label: "humans have flaws", description: "descriptions of how humans are flawed"},
    {feature: {atom: 25 * GPT4_ATOMS_PER_SHARD + 8, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
     label: "police reports, especially child safety", description: "safety incidents especially related to children"},
    {feature: {atom: 9 * GPT4_ATOMS_PER_SHARD + 44, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "price increases", description: "ends of phrases describing commodity/equity price increases"},
    {feature: {atom: 17 * GPT4_ATOMS_PER_SHARD + 33, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "ratification (multilingual)", description: "ends of phrases describing commodity/equity price increases"},
    {feature: {atom: 3 * GPT4_ATOMS_PER_SHARD + 421, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "would [...]", description: "conditionals (things that would be true)"},
    {feature: {atom: 63 * GPT4_ATOMS_PER_SHARD + 8, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "identification documents (multilingual)", description: "identification documents (multilingual)"},
    {feature: {atom: 0 * GPT4_ATOMS_PER_SHARD + 14, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "lightly incremented timestamps", description: "timestamps being lightly incremented with recurring formats"},
    {heading: 'Technical knowledge', heading_type: 'h3', feature: null, label: ''},
    {feature: {atom: 40 * GPT4_ATOMS_PER_SHARD + 42, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "machine learning training logs", description: "machine learning training logs"},
    {feature: {atom: 12 * GPT4_ATOMS_PER_SHARD + 47, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "onclick/onchange = function(this)", description: "onclick/onchange = function(this)"},
    {feature: {atom: 54 * GPT4_ATOMS_PER_SHARD + 23, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "edges (graph theory) and related concepts", description: "edges (graph theory) and related concepts"},
    {feature: {atom: 56 * GPT4_ATOMS_PER_SHARD + 12, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "algebraic rings", description: "algebraic rings"},
    {feature: {atom: 28 * GPT4_ATOMS_PER_SHARD + 47, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "adenosine/dopamine receptors", description: "adenosine/dopamine receptors"},
    {feature: {atom: 2 * GPT4_ATOMS_PER_SHARD + 601, autoencoder: AUTOENCODER_FAMILIES['v5_latelayer_postmlp'].get_ae({})},
      label: "blockchain vibes", description: "blockchain vibes"},


    {heading: 'GPT-2 small', heading_type: 'h4', feature: null, label: ''},
    {feature: {atom: 488432, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '2097152', num_active_features: '8'
    })}, label: "rhetorical questions", description: "rhetorical questions"},
    {feature: {atom: 2088200, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '2097152', num_active_features: '8'
    })}, label: "counting human casualties", description: "counting human casualties"},
    {feature: {atom: 1621560, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '2097152', num_active_features: '8'
    })}, label: "X and Y phrases", description: "X and -> Y"},
    {feature: {atom: 733, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '32768', num_active_features: '8'
    })}, label: "Patrick/Patty surname predictor", description: "Predicts surnames after Patrick"},
    {feature: {atom: 56907, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({ // similar to 33248
      num_features: '131072', num_active_features: '32'
    })}, label: "words in quotes", description: "predicts words in quotes"},
    {feature: {atom: 1605835, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '2097152', num_active_features: '8'
    })}, label: "these/those responsible things", description: "these/those, in a phrase where something is responsible for something"},
    {feature: {atom: 8040, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '8192', num_active_features: '32'
    })}, label: "2018 natural disasters", description: "2018 natural disasters"},
    {feature: {atom: 21464, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({ 
      num_features: '131072', num_active_features: '32'
    })}, label: "addition in code", description: "addition in code"},
    {feature: {atom: 66232, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({ 
      num_features: '131072', num_active_features: '32'
    })}, label: "function application", description: "function application"},
    {feature: {atom: 64464, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({ 
      num_features: '131072', num_active_features: '32'
    })}, label: "unclear/hidden things", description: "unclear/hidden things (top only)"},
    {feature: {atom: 10423, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '131072', num_active_features: '32'
    })}, label: "what the ...", description: "[who/what/when/where/why/how] the"},
    {heading: 'Safety relevant features (found via attribution methods)', heading_type: 'h3', feature: null, label: ''},
    {feature: {atom: 64840, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '131072', num_active_features: '32'
    })}, label: "profanity", description: "activates in order to output profanity"},
    {feature: {atom: 72185, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '131072', num_active_features: '32'
    })}, label: "erotic content", description: "erotic content"},
    {feature: {atom: 69134, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
      num_features: '131072', num_active_features: '32'
    })}, label: "[content warning] sexual abuse", description: "sexual abuse"},
    // {feature: {atom: 2, autoencoder: AUTOENCODER_FAMILIES['v5_l8_postmlp'].get_ae({
    //   num_features: '2097152', num_active_features: '8'
    // })}, label: "things being brought", description: "bring * -> together/back"},
  ]

  let [feature, setFeature] = useState({
    atom: 0, autoencoder: DEFAULT_AUTOENCODER
  })
  const handleClick = (click_feature: Feature) => {
    navigate(pathForFeature(click_feature))
  }

  return (
    <div className="flex flex-col" style={{'padding': '100px'}}>
      <h1 className="text-2xl font-bold mb-4">Welcome!  This is a viewer for sparse autoencoders features trained in <a href="todo">this paper</a> </h1>
      <h1>Pick a feature:</h1>
      <FeatureSelect
        init_feature={feature}
        onFeatureChange={(f: Feature) => setFeature(f)}
        onFeatureSubmit={(f: Feature) => navigate(pathForFeature(f))}
        show_go={true}
      />

      <div className="mt-4">
        <h2 className="text-xl font-bold mb-2">Interesting features:</h2>
        <div className="mb-10 flex-row">
          <div
            className="flex flex-flow flex-wrap"
          >
            {displayFeatures.map(({ heading, heading_type, feature, label, description }, j) => (
              heading ? <div style={{width: '100%'}} key={j}>
              {React.createElement(heading_type, {}, heading)}
              </div> : <button
                key={j}
                onClick={() => handleClick(feature)}
                style={{ width: 200 }}
                className="text-blue-500 hover:text-blue-700"
                title={description}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
