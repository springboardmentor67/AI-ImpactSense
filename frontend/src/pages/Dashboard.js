import React, { useState } from "react";
import { Doughnut } from "react-chartjs-2";
import "chart.js/auto";
import "./Dashboard.css";


const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

function Dashboard() {
  const [inputs, setInputs] = useState({
    magnitude: 7,
    depth: 10,
    cdi: 5,
    mmi: 5,
    sig: 0,
  });
  const [alert, setAlert] = useState("");
  const [dataset, setDataset] = useState([]);
  const [loading, setLoading] = useState(false);
  const [useSliders, setUseSliders] = useState(true);

  const handleChange = (e) => {
    const value = parseFloat(e.target.value);
    setInputs({ ...inputs, [e.target.id]: isNaN(value) ? 0 : value });
  };

  const predict = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      });
      if (!response.ok) throw new Error("Prediction failed");
      const data = await response.json();
      setAlert(data.alert);
    } catch (err) {
      setAlert(`❌ Error: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadDataset = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/dataset`);
      if (!response.ok) throw new Error("Dataset fetch failed");
      const data = await response.json();
      setDataset(data.rows);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const severityMap = {
    green: { value: 25, color: "green", text: "✅ Safe (Green)", bg: "#e6ffe6" },
    yellow: { value: 50, color: "gold", text: "⚠️ Caution (Yellow)", bg: "#fff9e6" },
    orange: { value: 75, color: "orange", text: "🔶 Warning (Orange)", bg: "#ffe6cc" },
    red: { value: 100, color: "red", text: "🚨 Danger (Red)", bg: "#ffe6e6" },
    error: { value: 0, color: "gray", text: "❌ Error fetching alert", bg: "#f0f0f0" },
  };
  const severity = severityMap[alert] || {
    value: 0,
    color: "#333",
    text: "Unknown",
    bg: "#ddd",
  };

  const chartData = {
    datasets: [
      {
        data: [severity.value, 100 - severity.value],
        backgroundColor: [severity.color, "#f4f4f4"],
      },
    ],
  };
  const chartOptions = {
    circumference: 180,
    rotation: -90,
    cutout: "70%",
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
  };

  return (
    <div>
      <div className="main-container">
        {/* Left Panel */}
        <div className="left-panel">
          <h2>Earthquake Alert Predictor</h2>

          {/* Toggle between sliders and input boxes */}
          <div className="toggle-container">
            <label>
              <input
                type="checkbox"
                checked={useSliders}
                onChange={() => setUseSliders(!useSliders)}
              />
              {useSliders ? "Using Sliders" : "Using Input Boxes"}
            </label>
          </div>

          <div className="grid">
            {["magnitude", "depth", "cdi", "mmi", "sig"].map((id) => (
              <div key={id} className="card">
                <label>
                  {id.toUpperCase()}: {inputs[id]}
                </label>

                {useSliders ? (
                  <input
                    type="range"
                    id={id}
                    min={
                      id === "depth"
                        ? 0
                        : id === "sig"
                        ? -150
                        : id === "magnitude"
                        ? 6
                        : 0
                    }
                    max={
                      id === "depth"
                        ? 700
                        : id === "sig"
                        ? 150
                        : id === "magnitude"
                        ? 10
                        : 9
                    }
                    step={id === "magnitude" ? 0.01 : 1}
                    value={inputs[id]}
                    onChange={handleChange}
                  />
                ) : (
                  <input
                    type="number"
                    id={id}
                    value={inputs[id]}
                    onChange={handleChange}
                  />
                )}
              </div>
            ))}
          </div>

          <button onClick={predict} disabled={loading}>
            {loading ? "Loading..." : "Predict Alert"}
          </button>

          <div
            className="result-box"
            style={{ background: severity.bg, color: severity.color }}
          >
            {severity.text}
          </div>
        </div>

        {/* Right Panel */}
        <div className="right-panel">
          <h2>📊 Dataset Preview</h2>
          <button onClick={loadDataset} disabled={loading}>
            {loading ? "Loading..." : "Load Dataset Preview"}
          </button>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Magnitude</th>
                  <th>Depth</th>
                  <th>CDI</th>
                  <th>MMI</th>
                  <th>SIG</th>
                  <th>Alert</th>
                </tr>
              </thead>
              <tbody>
                {dataset.map((row, idx) => (
                  <tr key={idx} className={`row-${row.alert}`}>
                    <td>{row.magnitude}</td>
                    <td>{row.depth}</td>
                    <td>{row.cdi}</td>
                    <td>{row.mmi}</td>
                    <td>{row.sig}</td>
                    <td>{row.alert}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div id="chartContainer">
            <Doughnut data={chartData} options={chartOptions} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
