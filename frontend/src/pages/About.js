import React from "react";
import "./About.css"
function About() {
  return (
    <div className="about-container">
      {/* Hero Banner */}
      <section className="about-hero">
        <h1>About QuakeAlert</h1>
        <p>Predicting earthquakes with clarity, insight, and technology.</p>
      </section>

      {/* Main Content */}
      <section className="about-content">
        <div className="about-card">
          <h2>🌍 Our Mission</h2>
          <p>
            QuakeAlert is designed to empower communities and students with
            real-time earthquake prediction tools. By combining seismic data
            with predictive modeling, we aim to make disaster preparedness
            accessible and actionable.
          </p>
        </div>

        <div className="about-card">
          <h2>⚡ Technology</h2>
          <p>
            Built with <strong>React</strong> for the frontend and <strong>Python</strong> for the backend,
            QuakeAlert integrates modern web development with machine learning
            models. The dashboard provides interactive sliders, severity
            visualization, and dataset previews for hands-on exploration.
          </p>
        </div>

        <div className="about-card">
          <h2>📊 Why It Matters</h2>
          <p>
            Earthquakes are unpredictable, but awareness saves lives. QuakeAlert
            demonstrates how technology can transform raw data into meaningful
            insights, helping faculty, students, and communities stay informed
            and prepared.
          </p>
        </div>
      </section>
    </div>
  );
}

export default About;
