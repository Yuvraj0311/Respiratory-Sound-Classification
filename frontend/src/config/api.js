// src/config/api.js

const getApiBaseUrl = () => {
  // Check for custom environment variable first
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '');
  }

  // Development mode (localhost)
  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }

  // Production fallback - set your deployed backend URL here
  return 'https://your-backend.onrender.com';  // Change this for production
};

export const API_BASE_URL = getApiBaseUrl();

export const getApiUrl = (endpoint = '') => {
  // Handle empty string case
  if (!endpoint) {
    return API_BASE_URL;
  }

  // Ensure endpoint starts with /
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${API_BASE_URL}${cleanEndpoint}`;
};

export const getAvatarUrl = (avatarPath) => {
  // Default avatar if none provided
  if (!avatarPath) {
    return 'https://ui-avatars.com/api/?name=Patient&size=150&background=4A90E2&color=ffffff';
  }

  // Return if already a full URL
  if (avatarPath.startsWith('http://') || avatarPath.startsWith('https://')) {
    return avatarPath;
  }

  // Build full URL from relative path
  const cleanPath = avatarPath.startsWith('/') ? avatarPath : `/${avatarPath}`;
  return `${API_BASE_URL}${cleanPath}`;
};

export default API_BASE_URL;
