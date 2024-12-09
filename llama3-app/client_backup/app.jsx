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
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Llama 2 Chat</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="text-red-500 border border-red-500 p-2 rounded">
            {error}
          </div>
        )}

        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          className="w-full p-2 border rounded"
          rows="4"
          placeholder="Enter your message..."
        />

        <button
          type="submit"
          disabled={loading || !message.trim()}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-blue-300"
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>

      {loading && (
        <div className="text-blue-500 mt-4">Processing your request...</div>
      )}

      {response && (
        <div className="mt-6">
          <h2 className="font-bold mb-2">Response:</h2>
          <p className="bg-gray-100 p-3 rounded-md border whitespace-pre-wrap">
            {response}
          </p>
        </div>
      )}
    </div>
  );
}
