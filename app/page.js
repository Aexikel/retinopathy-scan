"use client";
import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { jsPDF } from "jspdf";

export default function Home() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [mode, setMode] = useState('upload'); // 'upload' or 'camera'
  const [previewUrl, setPreviewUrl] = useState(null);
  
  const videoRef = useRef(null);
  const imgRef = useRef(null);

  const stages = [
    { name: 'No Diabetic Retinopathy', color: 'bg-green-500', icon: '👁️', risk: 'LOW', desc: 'No clinical signs of retinopathy detected in the retina.' },
    { name: 'Mild NPDR', color: 'bg-yellow-500', icon: '🟡', risk: 'MODERATE', desc: 'Microaneurysms only; small areas of swelling in blood vessels.' },
    { name: 'Moderate NPDR', color: 'bg-orange-500', icon: '🟠', risk: 'ELEVATED', desc: 'Blood vessels that nourish the retina may swell and distort.' },
    { name: 'Severe NPDR', color: 'bg-red-500', icon: '🔴', risk: 'HIGH', desc: 'Many blood vessels are blocked, depriving retina of blood supply.' },
    { name: 'Proliferative DR', color: 'bg-red-900', icon: '🌋', risk: 'CRITICAL', desc: 'Advanced stage where new, fragile blood vessels grow in the retina.' },
  ];

  useEffect(() => {
    async function load() {
      await tf.ready();
      const m = await tf.loadGraphModel('/tfjs_graph_model/model.json');
      setModel(m);
    }
    load();
  }, []);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setPrediction(null);
    }
  };

  const startCamera = async () => {
    setMode('camera');
    setPreviewUrl(null);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
  };

  const runAnalysis = async () => {
    const source = mode === 'camera' ? videoRef.current : imgRef.current;
    const tensor = tf.tidy(() => {
      return tf.browser.fromPixels(source).resizeNearestNeighbor([224, 224]).toFloat().div(255).expandDims();
    });
    const res = await model.predict(tensor);
    const data = await res.data();
    const idx = data.indexOf(Math.max(...data));
    setPrediction({ ...stages[idx], confidence: (Math.max(...data) * 100).toFixed(1) });
  };

  return (
    <div className="min-h-screen bg-white font-sans text-slate-900">
      {/* Blue Hero Header */}
      <header className="bg-gradient-to-b from-blue-600 to-blue-700 text-white py-16 px-4 text-center rounded-b-[3rem] shadow-xl">
        <div className="flex justify-center mb-4 text-4xl">📈</div>
        <h1 className="text-4xl font-bold mb-2">AI Diabetic Retinopathy Detection</h1>
        <p className="text-blue-100 max-w-xl mx-auto opacity-90">Advanced deep learning-based retinal image classification to identify severity stages in real-time.</p>
      </header>

      <main className="max-w-5xl mx-auto -mt-10 px-4">
        {/* Tab Switcher */}
        <div className="flex justify-center mb-8">
          <div className="bg-white p-1 rounded-full shadow-lg border flex gap-2">
            <button onClick={() => setMode('upload')} className={`px-6 py-2 rounded-full transition ${mode === 'upload' ? 'bg-blue-600 text-white' : 'text-slate-500'}`}>📁 Upload Image</button>
            <button onClick={startCamera} className={`px-6 py-2 rounded-full transition ${mode === 'camera' ? 'bg-blue-600 text-white' : 'text-slate-500'}`}>📷 Live Camera</button>
          </div>
        </div>

        {/* Input Card */}
        <div className="bg-white border-2 border-dashed border-blue-200 rounded-3xl p-8 mb-16 text-center shadow-sm">
          {mode === 'camera' ? (
            <video ref={videoRef} autoPlay className="mx-auto rounded-2xl h-64 bg-black mb-4" />
          ) : (
            <div className="h-64 flex flex-col items-center justify-center">
              {previewUrl ? (
                <img ref={imgRef} src={previewUrl} className="h-full rounded-2xl object-contain mb-4" />
              ) : (
                <div className="text-blue-500 mb-4 text-5xl">⬆️</div>
              )}
              <input type="file" onChange={handleFile} id="fileInput" hidden />
              <button onClick={() => document.getElementById('fileInput').click()} className="text-blue-600 font-semibold underline">Click to upload</button>
              <p className="text-slate-400 text-sm mt-1">High Resolution Fundus Image (PNG, JPG)</p>
            </div>
          )}
          <button 
            onClick={runAnalysis} 
            disabled={!model || (!previewUrl && mode==='upload')} 
            className="mt-6 px-12 py-3 bg-slate-900 text-white rounded-xl font-bold hover:bg-slate-800 disabled:opacity-50"
          >
            ⚡ Run Analysis
          </button>
        </div>

        {/* Results Section (Only shows when prediction exists) */}
        {prediction && (
          <div className="mb-16 animate-bounce-short">
            <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6 text-center">
                <h3 className="text-sm font-bold text-blue-600 uppercase tracking-widest">Diagnostic Result</h3>
                <div className="text-3xl font-black mt-1">{prediction.name}</div>
                <div className="text-slate-500 font-medium">Confidence: {prediction.confidence}%</div>
            </div>
          </div>
        )}

        {/* Info Grid */}
        <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-4 mb-20">
          {stages.map((s) => (
            <div key={s.name} className="bg-white border rounded-2xl p-5 shadow-sm hover:shadow-md transition">
              <div className={`${s.color} w-10 h-10 rounded-xl flex items-center justify-center text-white mb-4 shadow-sm`}>{s.icon}</div>
              <h4 className="font-bold text-sm mb-2">{s.name}</h4>
              <p className="text-xs text-slate-500 leading-relaxed mb-4">{s.desc}</p>
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Risk Level: <span className={s.name === prediction?.name ? 'text-blue-600' : ''}>{s.risk}</span></div>
            </div>
          ))}
        </div>
      </main>

      <footer className="bg-slate-900 text-slate-500 py-12 text-center text-sm">
        <p>© 2026 AI-Retina Healthcare Systems. All Rights Reserved.</p>
      </footer>
    </div>
  );
}