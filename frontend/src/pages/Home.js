import React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

function Home() {
  const navigate = useNavigate();

  return (
    <section className="hero">
      <h1>🌍 QuakeAlert</h1>
      <p>Real-time earthquake prediction and dataset insights.</p>
      <button onClick={() => navigate("/dashboard")}>Get Started</button>
    </section>
  );
}

export default Home;
