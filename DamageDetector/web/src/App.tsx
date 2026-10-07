import {  useEffect , useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Detection {
  class_name: string;
  confidence: number;
  box: Box;
}

interface PredictResponse {
  width: number;
  height: number;
  detections: Detection[];
}

function App() {
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [previewURL, setPreviewURL] = useState<string | null>(null) 

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    setPreviewURL(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || 'Error predicting damage');
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reach the server');
    } finally {
      setLoading(false);
    }
  }
  
  useEffect(() => {
  fetch(`${API_URL}/health`)
    .then((res) => res.json())
    .then((data) => console.log(data))
    .catch((err) => console.error('Error fetching health check:', err))
  }, []);

  return (
    <div className="App">
      <h1>Welcome to the Damage Detector</h1>
      <input
        type="file"
        accept="image/jpeg, image/png, image/webp"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {previewURL && <img src={previewURL} alt="Preview" style={{ maxWidth: '300px', marginTop: '20px' }} />}
      {result && (
        <div>
          <h2>Detections:</h2>
          <ul>
            {result.detections.map((detection, index) => (
              <li key={index}>
                Class: {detection.class_name}, Confidence: {(detection.confidence * 100).toFixed(2)}%, Box: ({detection.box.x1}, {detection.box.y1}) to ({detection.box.x2}, {detection.box.y2})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
