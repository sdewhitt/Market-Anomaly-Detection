import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

export default function Home() {
  const [prediction, setPrediction] = useState(null);

  const fetchPrediction = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/predict");
      const data = await response.json();
      setPrediction(data.predictions);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div>
        <h1>Predict with XGBoost</h1>
        <button onClick={fetchPrediction}>Get Prediction</button>
        {prediction && (
          <div>
            <h2>Prediction Results</h2>
            <p>{prediction}</p>
          </div>
        )}
      </div>
    </main>
  );
}
