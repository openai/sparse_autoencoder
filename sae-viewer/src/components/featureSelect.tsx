import FeatureInfo from "./components/featureInfo"
import React, { useEffect, useState, FormEvent } from "react"
import Welcome from "./welcome"
import { Feature } from "./types"
import { useState } from "react"

import { pathForFeature, DEFAULT_AUTOENCODER, SUBJECT_MODELS, AUTOENCODER_FAMILIES } from "../autoencoder_registry"
      
type FeatureSelectProps = {
  init_feature: Feature,
  onFeatureChange?: (feature: Feature) => void,
  onFeatureSubmit: (feature: Feature) => void,
}

export default function FeatureSelect({init_feature, onFeatureChange, onFeatureSubmit, show_go}: FeatureSelectProps) {
  let [feature, setFeature] = useState(init_feature);
  let family = AUTOENCODER_FAMILIES[feature.autoencoder.family];
  let [warningAcknowledged, setWarningAcknowledged] = useState<boolean>(localStorage.getItem('warningAcknowledged') === 'true');
  let changeFeature = (feature: Feature) => {
    onFeatureChange && onFeatureChange(feature);
    setFeature(feature);
  }
  // console.log('features', feature.autoencoder.num_features)
  const feelingLuckySubmit = () => {
    const atom = Math.floor(Math.random() * feature.autoencoder.num_features);
    const random_feature: Feature = {atom, autoencoder: feature.autoencoder}
    setFeature(random_feature);
    onFeatureSubmit(random_feature);
    return false
  }
  const acknowledgeWarning = () => {
    localStorage.setItem('warningAcknowledged', 'true');
    setWarningAcknowledged(true);
  }

  if (!warningAcknowledged) {
    return (<button
      onClick={acknowledgeWarning}
      className="border border-gray-300 rounded-md p-2 bg-red-200"
    >
      Note:  by clicking this button you acknowledge that the content of the documents are taken randomly from the internet, and may contain offensive or inappropriate content.
    </button>)
  }

  return (
    <>
    <h3>

    <b>Subject model</b> {" "} 
    <select value={feature.autoencoder.subject} onChange={(e) => {
      let family = Object.values(AUTOENCODER_FAMILIES).find((family) => (family.subject === e.target.value));
      changeFeature({
        atom: 0, autoencoder: family.get_ae(family.default_H(feature.autoencoder.H))
      })
    }}>
    {Object.values(SUBJECT_MODELS).map((subject_model) => (
      <option key={subject_model} value={subject_model}>{subject_model}</option>
    ))}
    </select>

    {" "}
    <br/>

    <b>Autoencoder</b> {" "} 
    <b>family</b> {" "} 
    <select value={feature.autoencoder.family} onChange={(e) => {
      let family = AUTOENCODER_FAMILIES[e.target.value];
      changeFeature({
        atom: 0, autoencoder: family.get_ae(family.default_H(feature.autoencoder.H))
      })
    }}>
    {Object.values(AUTOENCODER_FAMILIES).filter((family) => (family.subject === feature.autoencoder.subject)).map((family) => (
      <option key={family.name} value={family.name}>{AUTOENCODER_FAMILIES[family.name].label}</option>
    )) }
    </select>

    {
      AUTOENCODER_FAMILIES[feature.autoencoder.family].selectors.map((selector) => (
        <span key={selector.key}>
        {" "}
        {selector.label || selector.key} 
        <select value={feature.autoencoder.H[selector.key] || selector.values[0]} 
          onChange={(e) => {
            let family = AUTOENCODER_FAMILIES[feature.autoencoder.family];
            changeFeature({
              atom: 0, autoencoder: family.get_ae({...feature.autoencoder.H, [selector.key]: e.target.value})
            })
          }}
        >
          {selector.values.map((value) => (
            <option key={value} value={value}>{value}</option>
          ))}
        </select>
        </span>
      ))
    }
    {
      family.warning ? <span style={{color: 'red'}}>Note: {family.warning}</span> : null
    }

    <br/>
    <b>Feature</b>
    <input
          type="number"
          id="inputIndex"
          value={feature.atom}
          min={0}
          max={feature.autoencoder.num_features}
          style={{ width: 150, marginLeft: 10, marginRight: 10 }}
          onChange={(e) => (!isNaN(parseInt(e.target.value))) && changeFeature({...feature, atom: parseInt(e.target.value)})}
          className="border border-gray-300 rounded-md p-2"
        />
    {
      show_go && 
      <button
        onClick={
          (e: FormEvent) => {
            e.preventDefault()
            onFeatureSubmit(feature)
            return false
          }}
        className="border border-gray-300 rounded-md p-2"
        style={{ width: 200 }}
        disabled={!warningAcknowledged}
      >
        Go to feature {feature.atom}
      </button>
    }
    <br/>
      <button
        onClick={feelingLuckySubmit}
        className="border border-gray-300 rounded-md p-2"
        disabled={!warningAcknowledged}
      >
        I'm feeling lucky
      </button>


    <br/>

    </h3>
    </>
  )
}
