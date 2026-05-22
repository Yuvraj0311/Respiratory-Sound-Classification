import React, { useState } from "react";
import "../css/AuthPage.css";
import { useNavigate } from "react-router-dom";
import { GoogleLogin } from "@react-oauth/google";
import { jwtDecode } from "jwt-decode";
import { getApiUrl } from "../config/api";

const AuthPage = ({ onLoginSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: "",
    username: "",
    full_name: "",
    password: "",
    confirmPassword: "",
    username_or_email: "",
    login_password: "",
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError("");
  };


  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    if (!formData.email || !formData.username || !formData.full_name || !formData.password) {
      setError("All fields are required.");
      setLoading(false);
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters long.");
      setLoading(false);
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError("Please enter a valid email address.");
      setLoading(false);
      return;
    }

    try {
      const formDataToSend = new FormData();
      formDataToSend.append("email", formData.email);
      formDataToSend.append("username", formData.username);
      formDataToSend.append("full_name", formData.full_name);
      formDataToSend.append("password", formData.password);

      const response = await fetch(getApiUrl("/register"), {
        method: "POST",
        body: formDataToSend,
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess("Registration successful! You can now login.");
        setFormData({
          email: "",
          username: "",
          full_name: "",
          password: "",
          confirmPassword: "",
          username_or_email: "",
          login_password: "",
        });
        setTimeout(() => {
          setIsLogin(true);
          setSuccess("");
        }, 2000);
      } else {
        setError(data.detail || "Registration failed. Please try again.");
      }
    } catch (err) {
      console.error("Registration Error:", err);
      setError("Something went wrong. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

// ✅ FIXED handleLogin
const handleLogin = async (e) => {
  e.preventDefault();
  setError("");
  setLoading(true);

  if (!formData.username_or_email || !formData.login_password) {
    setError("Username/Email and password are required.");
    setLoading(false);
    return;
  }

  try {
    const formDataToSend = new FormData();
    formDataToSend.append("username_or_email", formData.username_or_email);
    formDataToSend.append("password", formData.login_password);

    const response = await fetch(getApiUrl("/login"), {
      method: "POST",
      body: formDataToSend,
    });

    const data = await response.json();

    if (response.ok) {
      // ✅ FIX 1: Store token with correct key name
      localStorage.setItem("token", data.access_token);

      // ✅ FIX 2: Store complete user object with patient_id
      const userToStore = {
        patient_id: data.user_info.patient_id,  // ✅ Most important!
        username: data.user_info.username,
        email: data.user_info.email,
        name: data.user_info.name,
        avatar: data.user_info.avatar,
        age_range: data.user_info.age_range,
        gender: data.user_info.gender,
      };

      localStorage.setItem("user", JSON.stringify(userToStore));

      // ✅ Debug log to verify
      console.log("✅ Login successful! Stored user:", userToStore);

      // Call the callback
      onLoginSuccess(userToStore);

      // Navigate to chat
      navigate("/chat");
    } else {
      setError(data.detail || "Login failed. Please check your credentials.");
    }
  } catch (err) {
    console.error("Login Error:", err);
    setError("Something went wrong. Please try again later.");
  } finally {
    setLoading(false);
  }
};

const handleGoogleSuccess = async (credentialResponse) => {
  setLoading(true);
  setError("");

  try {
    const decoded = jwtDecode(credentialResponse.credential);

    const googleUser = {
      email: decoded.email,
      name: decoded.name,
      picture: decoded.picture,
      google_id: decoded.sub,
    };

    const response = await fetch(getApiUrl("/google-login"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(googleUser),
    });

    const data = await response.json();

    if (response.ok) {
      localStorage.setItem("token", data.access_token);

      const userToStore = {
        patient_id: data.user_info.patient_id,
        username: data.user_info.username,
        email: data.user_info.email,
        name: data.user_info.name,
        avatar: data.user_info.avatar || decoded.picture,
        age_range: data.user_info.age_range,
        gender: data.user_info.gender,
      };

      localStorage.setItem("user", JSON.stringify(userToStore));

      //  Debug log to verify
      console.log("✅ Google login successful! Stored user:", userToStore);

      // Call the callback
      onLoginSuccess(userToStore);

      // Navigate to chat
      navigate("/chat");
    } else {
      setError(data.detail || "Google login failed.");
    }
  } catch (err) {
    console.error("Google Login Error:", err);
    setError("Something went wrong during Google login.");
  } finally {
    setLoading(false);
  }
};


  return (
    <div className="auth-container">
      <div className="auth-form-wrapper">
        <h2>{isLogin ? "Patient Login" : "Patient Registration"}</h2>
        {error && <p className="message error">{error}</p>}
        {success && <p className="message success">{success}</p>}

        {isLogin ? (
          <form onSubmit={handleLogin}>
            <div className="input-group">
              <label>Username or Email</label>
              <input
                type="text"
                name="username_or_email"
                value={formData.username_or_email}
                onChange={handleInputChange}
                placeholder="Enter your username or email"
                required
                disabled={loading}
              />
            </div>

            <div className="input-group">
              <label>Password</label>
              <input
                type="password"
                name="login_password"
                value={formData.login_password}
                onChange={handleInputChange}
                placeholder="Enter your password"
                required
                disabled={loading}
              />
            </div>

            <button type="submit" className="auth-button" disabled={loading}>
              {loading ? "Logging in..." : "Login"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleRegister}>
            <div className="input-group">
              <label>Email Address</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="your.email@example.com"
                required
                disabled={loading}
              />
            </div>

            <div className="input-group">
              <label>Username</label>
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleInputChange}
                placeholder="Choose a unique username"
                required
                disabled={loading}
              />
            </div>

            <div className="input-group">
              <label>Full Name</label>
              <input
                type="text"
                name="full_name"
                value={formData.full_name}
                onChange={handleInputChange}
                placeholder="John Doe"
                required
                disabled={loading}
              />
            </div>

            <div className="input-group">
              <label>Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Minimum 6 characters"
                required
                disabled={loading}
                minLength={6}
              />
            </div>

            <div className="input-group">
              <label>Confirm Password</label>
              <input
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleInputChange}
                placeholder="Re-enter your password"
                required
                disabled={loading}
                minLength={6}
              />
            </div>

            <button type="submit" className="auth-button" disabled={loading}>
              {loading ? "Registering..." : "Register"}
            </button>
          </form>
        )}

        <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            margin: "1rem 0",
            gap: "10px"
          }}>
            <div style={{ flex: 1, height: "1px", backgroundColor: "#ddd" }}></div>
            <span style={{ color: "#666", fontSize: "0.9rem" }}>OR</span>
            <div style={{ flex: 1, height: "1px", backgroundColor: "#ddd" }}></div>
          </div>

          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={() => setError("Google Sign-In failed.")}
            useOneTap
            text={isLogin ? "signin_with" : "signup_with"}
          />
        </div>

        <p className="toggle-text">
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <button
            onClick={() => {
              setIsLogin(!isLogin);
              setError("");
              setSuccess("");
              setFormData({
                email: "",
                username: "",
                full_name: "",
                password: "",
                confirmPassword: "",
                username_or_email: "",
                login_password: "",
              });
            }}
            className="toggle-button"
            disabled={loading}
          >
            {isLogin ? "Register" : "Login"}
          </button>
        </p>
      </div>
    </div>
  );
};

export default AuthPage;