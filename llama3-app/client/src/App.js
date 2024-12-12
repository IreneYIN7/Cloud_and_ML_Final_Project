import { useState } from "react";

export default function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      // Replace with your SageMaker endpoint URL
      const API_URL = process.env.REACT_APP_API_URL;
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message }), // Send the user input to the backend
      });

      if (!res.ok) {
        throw new Error("Failed to fetch response from backend.");
      }

      const data = await res.json();
      setResponse(data.generated_text); // Display the generated text from the backend
    } catch (err) {
      setError(err.message || "An error occurred while communicating with the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h1 className="chat-header">Llama3 Story Generator</h1>

      <form onSubmit={handleSubmit} className="chat-form">
        {error && <div className="error-message">{error}</div>}

        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          className="chat-textarea"
          rows="4"
          placeholder="Enter your message..."
        />

        <button
          type="submit"
          disabled={loading || !message.trim()}
          className="chat-button"
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>

      {loading && (
        <div className="loading-message">
          <div className="loading-dots" style={{animationDelay: "0s"}}></div>
          <div className="loading-dots" style={{animationDelay: "0.2s"}}></div>
          <div className="loading-dots" style={{animationDelay: "0.4s"}}></div>
        </div>
      )}

      {response && (
        <div className="response-container">
          <h2 className="response-header">Response</h2>
          <p className="response-text">{response}</p>
        </div>
      )}
    </div>
  );
}
