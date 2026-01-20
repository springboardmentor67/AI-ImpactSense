import React from "react";
import "./Contact.css";

function Contact() {
  return (
    <div className="contact-container">
      {/* Hero Banner */}
      <section className="contact-hero">
        <h1>Contact Us</h1>
        <p>We’d love to hear from you! Send us your feedback or questions.</p>
      </section>

      {/* Contact Form */}
      <section className="contact-form-section">
        <form className="contact-form">
          <label htmlFor="name">Name</label>
          <input type="text" id="name" placeholder="Enter your name" />

          <label htmlFor="email">Email</label>
          <input type="email" id="email" placeholder="Enter your email" />

          <label htmlFor="message">Message</label>
          <textarea id="message" rows="5" placeholder="Type your message here"></textarea>

          <button type="submit">Send Message</button>
        </form>
      </section>
    </div>
  );
}

export default Contact;
