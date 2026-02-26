"use client";
import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { jsPDF } from "jspdf";

export default function Home() {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isCamera, setIsCamera] = useState(false);
  
  const videoRef = useRef(null);
  const imgRef = useRef(null);

  const labels = ['No Diabetic Retinopathy', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'];
  const colors = ["text-green-600", "text-yellow-600", "text-orange-600", "text-red-600", "text-red-900"];

  // Load model on mount
  useEffect(() => {
    async function loadModel() {
      await tf.ready();
      // Notice the path starts with / because it's in the public folder
      const loadedModel = await tf.loadGraphModel('/tfjs_graph_model/model.json');
      setModel(loadedModel);
      setLoading(false);
    }
    loadModel();
  }, []);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    imgRef.current.src = url;
    imgRef.current.style.display = 'block';
    setIsCamera(false);
  };

  const startCamera = async () => {
    setIsCamera(true);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
  };

  const runAnalysis = async () => {
    const source = isCamera ? videoRef.current : imgRef.current;
    const tensor = tf.tidy(() => {
      return tf.browser.fromPixels(source)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();
    });

    const result = await model.predict(tensor);
    const data = await result.data();
    const classIdx = data.indexOf(Math.max(...data));
    setPrediction({
      label: labels[classIdx],
      color: colors[classIdx],
      confidence: (Math.max(...data) * 100).toFixed(1)
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white p-8 hidden md:flex flex-col">
        <h1 className="text-xl font-bold mb-10">👁️ RetinoScan AI</h1>
        <nav className="space-y-4 text-slate-400">
          <div className="text-white bg-slate-800 p-2 rounded">Dashboard</div>
          <div>Analytics</div>
          <div>Records</div>
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex-1 p-10 overflow-y-auto">
        <div className="flex justify-between items-center mb-10">
          <h2 className="text-3xl font-bold">Diagnostic Portal</h2>
          <span className={`px-4 py-1 rounded-full text-xs font-bold ${loading ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'}`}>
            {loading ? "Loading Model..." : "AI System Online"}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white p-6 rounded-2xl shadow-sm border">
            <h3 className="font-semibold mb-4">Acquisition</h3>
            <div className="aspect-video bg-black rounded-xl overflow-hidden mb-6">
              <video ref={videoRef} autoPlay className={isCamera ? "w-full h-full object-cover" : "hidden"} />
              <img ref={imgRef} className={!isCamera ? "w-full h-full object-contain" : "hidden"} />
            </div>
            <div className="flex gap-3">
              <input type="file" id="file" hidden onChange={handleUpload} />
              <button onClick={() => document.getElementById('file').click()} className="px-4 py-2 border rounded-lg hover:bg-slate-50">Upload</button>
              <button onClick={startCamera} className="px-4 py-2 border rounded-lg hover:bg-slate-50">Camera</button>
              <button onClick={runAnalysis} disabled={loading} className="px-6 py-2 bg-blue-600 text-white rounded-lg font-bold disabled:bg-slate-300">Run Analysis</button>
            </div>
          </div>

          {prediction && (
            <div className="bg-white p-8 rounded-2xl shadow-sm border animate-in fade-in slide-in-from-bottom-4">
              <h3 className="font-semibold mb-2">Result</h3>
              <div className={`text-4xl font-black ${prediction.color} mb-2`}>{prediction.label}</div>
              <p className="text-slate-500 mb-8">Confidence Score: {prediction.confidence}%</p>
              <button className="w-full py-3 bg-slate-900 text-white rounded-xl font-bold">Download Report</button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}