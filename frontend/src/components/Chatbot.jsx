import React, { useState, useRef, useEffect } from 'react';
import '../css/Chatbot.css';
import { FiUser, FiCpu, FiSend, FiPaperclip, FiXCircle } from 'react-icons/fi';
import { getApiUrl } from '../config/api';

const Chatbot = ({ user }) => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! I'm A.I.R.A., your AI health assistant. You can ask me questions or attach an audio file for respiratory analysis.",
      sender: "bot"
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac'];
    const validExtensions = /\.(wav|mp3|flac)$/i;

    if (!validTypes.includes(file.type) && !file.name.match(validExtensions)) {
      alert('Please select a valid audio file (WAV, MP3, or FLAC)');
      return;
    }

    setAudioFile(file);
  };

  // Chatbot.jsx - COMPLETE FIX

// Chatbot.jsx - COMPLETE FIX

const handleSend = async () => {
  if ((!input.trim() && !audioFile) || loading || !user) return;

  const userMessage = input.trim()
    ? { text: input, sender: "user" }
    : { text: "🎤 Audio file attached", sender: "user" };

  setMessages((prev) => [...prev, userMessage]);

  const messageToSend = input.trim() || "Please analyze this audio file";
  const fileToSend = audioFile;

  setInput("");
  setAudioFile(null);
  setLoading(true);

  try {
    let audioResult = null;

    // If audio file exists, analyze it first
    if (fileToSend) {
      const formData = new FormData();

      formData.append("patient_id", user.patient_id || user.patientId);
      formData.append("user_query", messageToSend);
      formData.append("file", fileToSend);

      const audioResponse = await fetch(`${getApiUrl()}/api/analyze-audio`, {
        method: "POST",
        body: formData,
      });

      if (!audioResponse.ok) {
        const errorData = await audioResponse.json();
        throw new Error(errorData.detail || "Audio analysis failed");
      }

      const audioData = await audioResponse.json();
      audioResult = audioData;

      const analysisInfo = `🔬 Audio Analysis: ${audioResult.disease} | Confidence: ${(
        audioResult.confidence * 100
      ).toFixed(1)}%`;

      setMessages((prev) => [...prev, { text: analysisInfo, sender: "bot", isInfo: true }]);
    }

    // Send chat request with optional audio result
    const chatResponse = await fetch(`${getApiUrl()}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        patient_id: user.patient_id || user.patientId,
        query: messageToSend,
        audio_result: audioResult,
      }),
    });

    if (!chatResponse.ok) {
      const errorData = await chatResponse.json();
      throw new Error(errorData.detail || "Failed to get AI response");
    }

    const chatData = await chatResponse.json();

    const botMessage = {
      text: chatData.response,
      sender: "bot",
    };

    setMessages((prev) => [...prev, botMessage]);
  } catch (error) {
    console.error("API Error:", error);
    const errorMessage = {
      text: `Sorry, an error occurred: ${error.message}. Please try again.`,
      sender: "bot",
    };
    setMessages((prev) => [...prev, errorMessage]);
  } finally {
    setLoading(false);
  }
};


  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h2>⚕️ Welcome to A.I.R.A., {user.username}</h2>
        <p>Your Personal AI Health Companion</p>
      </div>

      <div className="chatbot-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message-wrapper ${message.sender}`}>
            {!message.isInfo && (
              <div className="message-icon">
                {message.sender === "user" ? <FiUser /> : <FiCpu />}
              </div>
            )}
            <div className={`message-bubble ${message.isInfo ? 'info' : ''}`}>
              {message.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-wrapper bot">
            <div className="message-icon"><FiCpu /></div>
            <div className="message-bubble thinking">Analyzing...</div>
          </div>
        )}

        <div ref={messagesEndRef}></div>
      </div>

      <div className="chatbot-input-area">
        {audioFile && (
          <div className="file-preview">
            <span>🎙️ {audioFile.name}</span>
            <button onClick={() => setAudioFile(null)} type="button">
              <FiXCircle />
            </button>
          </div>
        )}

        <div className="chatbot-input-wrapper">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="audio/wav,audio/mpeg,audio/mp3,audio/flac,.wav,.mp3,.flac"
            style={{ display: 'none' }}
          />

          <button
            className="attach-button"
            onClick={() => fileInputRef.current.click()}
            disabled={loading}
            type="button"
            title="Attach audio file"
          >
            <FiPaperclip />
          </button>

          <input
            type="text"
            className="chatbot-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyPress={e => e.key === "Enter" && handleSend()}
            placeholder="Ask a question or attach an audio file..."
            disabled={loading}
          />

          <button
            className="send-button"
            onClick={handleSend}
            disabled={loading || (!input.trim() && !audioFile)}
            type="button"
            title="Send message"
          >
            <FiSend />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;