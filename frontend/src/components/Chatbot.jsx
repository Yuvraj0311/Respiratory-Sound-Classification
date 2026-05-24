import React, { useState, useRef, useEffect } from 'react';
import '../css/Chatbot.css';
import {
  FiUser,
  FiCpu,
  FiSend,
  FiPaperclip,
  FiXCircle,
  FiVolume2,
  FiImage,
} from 'react-icons/fi';
import { getApiUrl } from '../config/api';

const AUDIO_TYPES = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/x-wav'];
const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
const AUDIO_EXTENSION = /\.(wav|mp3|flac)$/i;
const IMAGE_EXTENSION = /\.(jpe?g|png|bmp|tiff?|webp)$/i;

const detectAttachmentType = (file) => {
  if (AUDIO_TYPES.includes(file.type) || AUDIO_EXTENSION.test(file.name)) {
    return 'audio';
  }
  if (IMAGE_TYPES.includes(file.type) || IMAGE_EXTENSION.test(file.name)) {
    return 'xray';
  }
  return null;
};

const formatConfidence = (confidence = 0) => `${(confidence * 100).toFixed(1)}%`;

const Chatbot = ({ user }) => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! I'm A.I.R.A., your AI health assistant. You can ask me questions or attach a lung sound recording or chest X-ray image for analysis.",
      sender: 'bot',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [attachment, setAttachment] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const conversationIdRef = useRef(
    window.crypto?.randomUUID
      ? window.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const type = detectAttachmentType(file);
    if (!type) {
      alert('Please select a valid audio file (WAV, MP3, FLAC) or X-ray image (JPG, PNG, BMP, TIFF, WEBP).');
      event.target.value = '';
      return;
    }

    setAttachment({ file, type });
    event.target.value = '';
  };

  const getAttachmentLabel = (item) => {
    if (!item) return '';
    return item.type === 'audio'
      ? `Audio file attached: ${item.file.name}`
      : `Chest X-ray image attached: ${item.file.name}`;
  };

  const handleSend = async () => {
    if ((!input.trim() && !attachment) || loading || !user) return;

    const attachmentLabel = getAttachmentLabel(attachment);
    const userMessage = input.trim()
      ? {
          text: attachmentLabel ? `${input.trim()} (${attachmentLabel})` : input.trim(),
          sender: 'user',
        }
      : { text: attachmentLabel, sender: 'user' };

    setMessages((prev) => [...prev, userMessage]);

    const messageToSend =
      input.trim() ||
      (attachment?.type === 'xray'
        ? 'Please analyze this chest X-ray image'
        : 'Please analyze this audio file');
    const fileToSend = attachment?.file;
    const attachmentType = attachment?.type;
    const recentUserMessages = [
      ...messages
        .filter((message) => message.sender === 'user')
        .slice(-5)
        .map((message) => message.text),
      userMessage.text,
    ];

    setInput('');
    setAttachment(null);
    setLoading(true);

    try {
      let audioResult = null;
      let xrayResult = null;

      if (fileToSend) {
        const formData = new FormData();
        formData.append('patient_id', user.patient_id || user.patientId);
        formData.append('user_query', messageToSend);
        formData.append('file', fileToSend);

        const endpoint = attachmentType === 'audio' ? '/api/analyze-audio' : '/api/analyze-xray';
        const analysisResponse = await fetch(`${getApiUrl()}${endpoint}`, {
          method: 'POST',
          body: formData,
        });

        if (!analysisResponse.ok) {
          const errorData = await analysisResponse.json();
          throw new Error(errorData.detail || 'File analysis failed');
        }

        const analysisData = await analysisResponse.json();

        if (attachmentType === 'audio') {
          audioResult = analysisData;
          setMessages((prev) => [
            ...prev,
            {
              text: `Audio Analysis: ${audioResult.disease} | Confidence: ${formatConfidence(audioResult.confidence)}`,
              sender: 'bot',
              isInfo: true,
            },
          ]);
        } else {
          xrayResult = analysisData;
          setMessages((prev) => [
            ...prev,
            {
              text: `X-ray Analysis: ${xrayResult.finding} | Confidence: ${formatConfidence(xrayResult.confidence)}`,
              sender: 'bot',
              isInfo: true,
            },
          ]);
        }
      }

      const chatResponse = await fetch(`${getApiUrl()}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: user.patient_id || user.patientId,
          conversation_id: conversationIdRef.current,
          query: messageToSend,
          recent_messages: recentUserMessages,
          audio_result: audioResult,
          xray_result: xrayResult,
        }),
      });

      if (!chatResponse.ok) {
        const errorData = await chatResponse.json();
        throw new Error(errorData.detail || 'Failed to get AI response');
      }

      const chatData = await chatResponse.json();
      setMessages((prev) => [...prev, { text: chatData.response, sender: 'bot' }]);
    } catch (error) {
      console.error('API Error:', error);
      setMessages((prev) => [
        ...prev,
        {
          text: `Sorry, an error occurred: ${error.message}. Please try again.`,
          sender: 'bot',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h2>Welcome to A.I.R.A., {user.username}</h2>
        <p>Your Personal AI Health Companion</p>
      </div>

      <div className="chatbot-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message-wrapper ${message.sender}`}>
            {!message.isInfo && (
              <div className="message-icon">
                {message.sender === 'user' ? <FiUser /> : <FiCpu />}
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
        {attachment && (
          <div className="file-preview">
            <span>
              {attachment.type === 'audio' ? <FiVolume2 /> : <FiImage />}
              {attachment.file.name}
            </span>
            <button onClick={() => setAttachment(null)} type="button" title="Remove attachment">
              <FiXCircle />
            </button>
          </div>
        )}

        <div className="chatbot-input-wrapper">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="audio/wav,audio/mpeg,audio/mp3,audio/flac,.wav,.mp3,.flac,image/jpeg,image/png,image/bmp,image/tiff,image/webp,.jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp"
            style={{ display: 'none' }}
          />

          <button
            className="attach-button"
            onClick={() => fileInputRef.current.click()}
            disabled={loading}
            type="button"
            title="Attach audio or chest X-ray image"
          >
            <FiPaperclip />
          </button>

          <input
            type="text"
            className="chatbot-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question or attach audio/X-ray image..."
            disabled={loading}
          />

          <button
            className="send-button"
            onClick={handleSend}
            disabled={loading || (!input.trim() && !attachment)}
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
