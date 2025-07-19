import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import axios from "axios";

// Phase 4: OAuth Integration
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import AppleSignin from 'react-apple-signin-auth';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const App = () => {
  const [apiKey, setApiKey] = useState("");
  const [therapyArea, setTherapyArea] = useState("");
  const [productName, setProductName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStates, setLoadingStates] = useState({
    analysis: false,
    funnel: false,
    competitive: false,
    scenarios: false,
    trials: false,
    search: false,
    company: false,
    ensemble: false,
    financial: false,
    timeline: false,
    template: false,
    visualization: false,
    export: false,
    rwe: false,
    market_access: false,
    predictive: false,
    dashboard: false,
    login: false,
    register: false,
    payment: false,
    workflows: false
  });
  const [analysis, setAnalysis] = useState(null);
  const [funnel, setFunnel] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("analysis");
  const [savedAnalyses, setSavedAnalyses] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [competitiveData, setCompetitiveData] = useState(null);
  const [scenarioModels, setScenarioModels] = useState(null);
  const [clinicalTrials, setClinicalTrials] = useState([]);
  const [exportLoading, setExportLoading] = useState(false);
  const [perplexityKey, setPerplexityKey] = useState("");
  const [realTimeSearch, setRealTimeSearch] = useState("");
  const [perplexityResults, setPerplexityResults] = useState(null);
  const [companyIntelligence, setCompanyIntelligence] = useState(null);
  const [ensembleResult, setEnsembleResult] = useState(null);
  const [geminiKey, setGeminiKey] = useState("");
  const [financialModel, setFinancialModel] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [customTemplate, setCustomTemplate] = useState(null);
  const [advancedViz, setAdvancedViz] = useState(null);
  
  // Phase 3: Real-World Evidence & Market Access state variables
  const [realWorldEvidence, setRealWorldEvidence] = useState(null);
  const [marketAccessIntel, setMarketAccessIntel] = useState(null);
  const [predictiveAnalytics, setPredictiveAnalytics] = useState(null);
  const [phase3Dashboard, setPhase3Dashboard] = useState(null);

  // Phase 4: Authentication & User Management state variables
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [showPaymentModal, setShowPaymentModal] = useState(false);
  const [subscriptionPlans, setSubscriptionPlans] = useState([]);
  const [userWorkflows, setUserWorkflows] = useState([]);
  const [executiveDashboard, setExecutiveDashboard] = useState(null);

  const setLoadingState = (operation, isLoading) => {
    setLoadingStates(prev => ({
      ...prev,
      [operation]: isLoading
    }));
  };

  // Component for rendering Plotly charts
  const PlotlyChart = ({ data, id }) => {
    const chartRef = useRef(null);
    
    useEffect(() => {
      if (data && chartRef.current && window.Plotly) {
        try {
          const plotData = JSON.parse(data);
          window.Plotly.newPlot(chartRef.current, plotData.data, plotData.layout, {
            responsive: true,
            displayModeBar: false
          });
        } catch (error) {
          console.error('Error rendering Plotly chart:', error);
        }
      }
    }, [data]);

    useEffect(() => {
      // Load Plotly if not already loaded
      if (!window.Plotly) {
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
        script.async = true;
        script.onload = () => {
          if (data && chartRef.current) {
            try {
              const plotData = JSON.parse(data);
              window.Plotly.newPlot(chartRef.current, plotData.data, plotData.layout, {
                responsive: true,
                displayModeBar: false
              });
            } catch (error) {
              console.error('Error rendering Plotly chart:', error);
            }
          }
        };
        document.head.appendChild(script);
      }
    }, []);

    return <div ref={chartRef} id={id} style={{ width: '100%', height: '500px' }} />;
  };

  useEffect(() => {
    const testConnection = async () => {
      try {
        await axios.get(`${API}/`);
        console.log("API connection successful");
        loadSavedAnalyses();
      } catch (e) {
        console.error("API connection failed:", e);
      }
    };
    testConnection();
  }, []);

  const loadSavedAnalyses = async () => {
    try {
      const response = await axios.get(`${API}/analyses`);
      setSavedAnalyses(response.data);
    } catch (error) {
      console.error("Failed to load analyses:", error);
    }
  };

  const handleAnalyzeTherapy = async () => {
    if (!apiKey.trim()) {
      setError("Please enter your Anthropic API key");
      return;
    }
    if (!therapyArea.trim()) {
      setError("Please enter a therapy area");
      return;
    }

    setLoadingState('analysis', true);
    setIsLoading(true);
    setError("");
    setFunnel(null);
    setCompetitiveData(null);
    setScenarioModels(null);

    try {
      const response = await axios.post(`${API}/analyze-therapy`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        api_key: apiKey
      });

      setAnalysis(response.data);
      setActiveTab("analysis");
      await loadSavedAnalyses();
    } catch (error) {
      console.error("Analysis error:", error);
      setError(error.response?.data?.detail || "Analysis failed. Please check your API key and try again.");
    } finally {
      setLoadingState('analysis', false);
      setIsLoading(false);
    }
  };

  const handleGenerateFunnel = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('funnel', true);
    setError("");

    try {
      const response = await axios.post(`${API}/generate-funnel`, {
        therapy_area: therapyArea,
        analysis_id: analysis.id,
        api_key: apiKey
      });

      setFunnel(response.data);
      setActiveTab("funnel");
    } catch (error) {
      console.error("Funnel generation error:", error);
      setError(error.response?.data?.detail || "Funnel generation failed. Please try again.");
    } finally {
      setLoadingState('funnel', false);
    }
  };

  const handleCompetitiveAnalysis = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('competitive', true);
    setError("");

    try {
      const response = await axios.post(`${API}/competitive-analysis`, {
        therapy_area: therapyArea,
        analysis_id: analysis.id,
        api_key: apiKey
      });

      setCompetitiveData(response.data.competitive_landscape);
      setActiveTab("competitive");
    } catch (error) {
      console.error("Competitive analysis error:", error);
      setError(error.response?.data?.detail || "Competitive analysis failed. Please try again.");
    } finally {
      setLoadingState('competitive', false);
    }
  };

  const handleScenarioModeling = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('scenarios', true);
    setError("");

    try {
      const response = await axios.post(`${API}/scenario-modeling`, {
        therapy_area: therapyArea,
        analysis_id: analysis.id,
        scenarios: ["optimistic", "realistic", "pessimistic"],
        api_key: apiKey
      });

      setScenarioModels(response.data.scenario_models);
      setActiveTab("scenarios");
    } catch (error) {
      console.error("Scenario modeling error:", error);
      setError(error.response?.data?.detail || "Scenario modeling failed. Please try again.");
    } finally {
      setLoadingState('scenarios', false);
    }
  };

  const handleCreateFinancialModel = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('financial', true);
    setError("");

    try {
      const response = await axios.post(`${API}/financial-model`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        analysis_id: analysis.id,
        discount_rate: 0.12,
        peak_sales_estimate: 1000,
        patent_expiry_year: 2035,
        launch_year: 2025,
        ramp_up_years: 5,
        monte_carlo_iterations: 1000,
        api_key: apiKey
      });

      setFinancialModel(response.data);
      setActiveTab("financial");
    } catch (error) {
      console.error("Financial model error:", error);
      setError(error.response?.data?.detail || "Financial modeling failed. Please try again.");
    } finally {
      setLoadingState('financial', false);
    }
  };

  const handleCreateTimeline = async () => {
    if (!analysis || !perplexityKey.trim()) {
      setError("Please complete analysis and enter Perplexity API key first");
      return;
    }

    setLoadingState('timeline', true);
    setError("");

    try {
      const response = await axios.post(`${API}/timeline`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        analysis_id: analysis.id,
        include_competitive_milestones: true,
        api_key: perplexityKey
      });

      setTimeline(response.data);
      setActiveTab("timeline");
    } catch (error) {
      console.error("Timeline error:", error);
      setError(error.response?.data?.detail || "Timeline generation failed. Please try again.");
    } finally {
      setLoadingState('timeline', false);
    }
  };

  const handleCreateTemplate = async (templateType = "therapy_specific") => {
    if (!perplexityKey.trim()) {
      setError("Please enter Perplexity API key for template generation");
      return;
    }

    setLoadingState('template', true);
    setError("");

    try {
      const response = await axios.post(`${API}/custom-template`, {
        template_type: templateType,
        therapy_area: therapyArea,
        region: "Global",
        api_key: perplexityKey
      });

      setCustomTemplate(response.data);
      setActiveTab("template");
    } catch (error) {
      console.error("Template error:", error);
      setError(error.response?.data?.detail || "Template generation failed. Please try again.");
    } finally {
      setLoadingState('template', false);
    }
  };

  const handleAdvancedVisualization = async (vizType, dataSource) => {
    if (!analysis) {
      setError("Please complete analysis first");
      return;
    }

    setLoadingState('visualization', true);
    setError("");

    try {
      const response = await axios.post(`${API}/advanced-visualization`, null, {
        params: {
          visualization_type: vizType,
          data_source: dataSource,
          analysis_id: analysis.id
        }
      });

      setAdvancedViz(response.data);
      setActiveTab("visualization");
    } catch (error) {
      console.error("Visualization error:", error);
      setError(error.response?.data?.detail || "Visualization generation failed. Please try again.");
    } finally {
      setLoadingState('visualization', false);
    }
  };

  const handleEnsembleAnalysis = async () => {
    if (!apiKey.trim()) {
      setError("Please enter your Claude API key");
      return;
    }
    if (!perplexityKey.trim()) {
      setError("Please enter your Perplexity API key for ensemble analysis");
      return;
    }
    if (!therapyArea.trim()) {
      setError("Please enter a therapy area");
      return;
    }

    setLoadingState('ensemble', true);
    setError("");

    try {
      const response = await axios.post(`${API}/ensemble-analysis`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        analysis_type: "comprehensive",
        claude_api_key: apiKey,
        perplexity_api_key: perplexityKey,
        gemini_api_key: geminiKey || null,
        use_gemini: geminiKey.trim().length > 0,
        confidence_threshold: 0.7
      });

      setEnsembleResult(response.data);
      setActiveTab("ensemble");
    } catch (error) {
      console.error("Ensemble analysis error:", error);
      setError(error.response?.data?.detail || "Ensemble analysis failed. Please check your API keys.");
    } finally {
      setLoadingState('ensemble', false);
    }
  };

  const handleCompanyIntelligence = async () => {
    if (!perplexityKey.trim()) {
      setError("Please enter your Perplexity API key for company intelligence");
      return;
    }
    if (!productName.trim()) {
      setError("Please enter a product name for company intelligence");
      return;
    }

    setLoadingState('company', true);
    setError("");

    try {
      const response = await axios.post(`${API}/company-intelligence`, {
        product_name: productName,
        therapy_area: therapyArea,
        api_key: perplexityKey,
        include_competitors: true
      });

      setCompanyIntelligence(response.data);
      setActiveTab("company");
    } catch (error) {
      console.error("Company intelligence error:", error);
      setError(error.response?.data?.detail || "Company intelligence failed. Please check your Perplexity API key.");
    } finally {
      setLoadingState('company', false);
    }
  };

  const handleRealTimeSearch = async () => {
    if (!perplexityKey.trim()) {
      setError("Please enter your Perplexity API key");
      return;
    }
    if (!realTimeSearch.trim()) {
      setError("Please enter a search query");
      return;
    }

    setLoadingState('search', true);
    setError("");

    try {
      const response = await axios.post(`${API}/perplexity-search`, {
        query: realTimeSearch,
        api_key: perplexityKey,
        search_focus: "pharmaceutical"
      });

      setPerplexityResults(response.data);
      setActiveTab("search");
    } catch (error) {
      console.error("Real-time search error:", error);
      setError(error.response?.data?.detail || "Real-time search failed. Please check your Perplexity API key.");
    } finally {
      setLoadingState('search', false);
    }
  };

  const handleEnhancedCompetitive = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('competitive', true);
    setError("");

    try {
      const response = await axios.post(`${API}/enhanced-competitive-analysis`, {
        therapy_area: therapyArea,
        analysis_id: analysis.id,
        api_key: apiKey  // Using Claude key for both - can be separated later
      });

      setCompetitiveData(response.data.competitive_landscape);
      setActiveTab("competitive");
    } catch (error) {
      console.error("Enhanced competitive analysis error:", error);
      setError(error.response?.data?.detail || "Enhanced competitive analysis failed. Please try again.");
    } finally {
      setLoadingState('competitive', false);
    }
  };

  const handleSearchTrials = async () => {
    if (!therapyArea.trim()) {
      setError("Please enter a therapy area first");
      return;
    }

    setLoadingState('trials', true);
    setError("");

    try {
      const response = await axios.get(`${API}/search/clinical-trials`, {
        params: { therapy_area: therapyArea }
      });
      setClinicalTrials(response.data.trials || []);
      setActiveTab("trials");
    } catch (error) {
      console.error("Clinical trials search error:", error);
      setError("Failed to search clinical trials");
    } finally {
      setLoadingState('trials', false);
    }
  };

  const handleExport = async (exportType) => {
    if (!analysis) {
      setError("No analysis to export");
      return;
    }

    setExportLoading(true);
    try {
      const response = await axios.post(`${API}/export`, {
        analysis_id: analysis.id,
        export_type: exportType
      });

      if (response.data.status === "success") {
        // Create download link
        const byteCharacters = atob(response.data.data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { 
          type: exportType === 'pdf' ? 'application/pdf' : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
        });

        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = response.data.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error("Export error:", error);
      setError("Export failed. Please try again.");
    } finally {
      setExportLoading(false);
    }
  };

  // Phase 3: Real-World Evidence Integration Handler Functions
  const handleRealWorldEvidence = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('rwe', true);
    setError("");

    try {
      const response = await axios.post(`${API}/real-world-evidence`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        analysis_type: "comprehensive",
        data_sources: ["registries", "claims", "ehr", "patient_outcomes"],
        api_key: apiKey
      });

      setRealWorldEvidence(response.data);
      setActiveTab("rwe");
    } catch (error) {
      console.error("Real-world evidence error:", error);
      setError(error.response?.data?.detail || "Real-world evidence analysis failed. Please try again.");
    } finally {
      setLoadingState('rwe', false);
    }
  };

  const handleMarketAccessIntelligence = async () => {
    if (!analysis || !perplexityKey.trim()) {
      setError("Please complete analysis and enter Perplexity API key first");
      return;
    }

    setLoadingState('market_access', true);
    setError("");

    try {
      const response = await axios.post(`${API}/market-access-intelligence`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        target_markets: ["US", "EU5", "Japan"],
        analysis_depth: "comprehensive",
        api_key: perplexityKey
      });

      setMarketAccessIntel(response.data);
      setActiveTab("market_access");
    } catch (error) {
      console.error("Market access intelligence error:", error);
      setError(error.response?.data?.detail || "Market access analysis failed. Please try again.");
    } finally {
      setLoadingState('market_access', false);
    }
  };

  const handlePredictiveAnalytics = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setLoadingState('predictive', true);
    setError("");

    try {
      const response = await axios.post(`${API}/predictive-analytics`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        forecast_horizon: 10,
        model_type: "ml_enhanced",
        include_rwe: true,
        api_key: apiKey
      });

      setPredictiveAnalytics(response.data);
      setActiveTab("predictive");
    } catch (error) {
      console.error("Predictive analytics error:", error);
      setError(error.response?.data?.detail || "Predictive analytics failed. Please try again.");
    } finally {
      setLoadingState('predictive', false);
    }
  };

  const handlePhase3Dashboard = async () => {
    setLoadingState('dashboard', true);
    setError("");

    try {
      const response = await axios.get(`${API}/phase3-dashboard`);
      setPhase3Dashboard(response.data);
      setActiveTab("dashboard");
    } catch (error) {
      console.error("Phase 3 dashboard error:", error);
      setError(error.response?.data?.detail || "Dashboard data retrieval failed. Please try again.");
    } finally {
      setLoadingState('dashboard', false);
    }
  };

  const loadAnalysis = async (analysisId) => {
    try {
      const response = await axios.get(`${API}/analysis/${analysisId}`);
      setAnalysis(response.data.analysis);
      setFunnel(response.data.funnel);
      setTherapyArea(response.data.analysis.therapy_area);
      setProductName(response.data.analysis.product_name || "");
      setShowHistory(false);
      setActiveTab("analysis");
    } catch (error) {
      console.error("Failed to load analysis:", error);
      setError("Failed to load saved analysis");
    }
  };

  // Phase 4: Authentication & Payment Handler Functions
  const handleRegister = async (userData) => {
    setLoadingState('register', true);
    setError("");

    try {
      const response = await axios.post(`${API}/auth/register`, userData);
      setShowRegisterModal(false);
      setShowLoginModal(true);
      // Show success message
      setError("Registration successful! Please log in.");
    } catch (error) {
      console.error("Registration error:", error);
      setError(error.response?.data?.detail || "Registration failed. Please try again.");
    } finally {
      setLoadingState('register', false);
    }
  };

  const handleLogin = async (loginData) => {
    setLoadingState('login', true);
    setError("");

    try {
      const response = await axios.post(`${API}/auth/login`, loginData);
      const { access_token, user } = response.data;
      
      // Store token and user info
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      
      // Set authentication state
      setIsAuthenticated(true);
      setCurrentUser(user);
      setShowLoginModal(false);
      
      // Configure axios default headers
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
    } catch (error) {
      console.error("Login error:", error);
      setError(error.response?.data?.detail || "Login failed. Please try again.");
    } finally {
      setLoadingState('login', false);
    }
  };

  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`);
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      // Clear local storage and state
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setIsAuthenticated(false);
      setCurrentUser(null);
      delete axios.defaults.headers.common['Authorization'];
    }
  };

  const loadSubscriptionPlans = async () => {
    try {
      const response = await axios.get(`${API}/subscriptions/plans`);
      setSubscriptionPlans(response.data.plans);
    } catch (error) {
      console.error("Failed to load subscription plans:", error);
    }
  };

  const handlePayment = async (packageId) => {
    setLoadingState('payment', true);
    setError("");

    try {
      const originUrl = window.location.origin;
      const response = await axios.post(`${API}/payments/checkout/session`, {
        package_id: packageId
      });

      // Redirect to Stripe Checkout
      window.location.href = response.data.url;
      
    } catch (error) {
      console.error("Payment error:", error);
      setError(error.response?.data?.detail || "Payment setup failed. Please try again.");
    } finally {
      setLoadingState('payment', false);
    }
  };

  const checkPaymentStatus = async (sessionId) => {
    let attempts = 0;
    const maxAttempts = 5;
    const pollInterval = 2000; // 2 seconds

    const poll = async () => {
      if (attempts >= maxAttempts) {
        setError("Payment status check timed out. Please check your email for confirmation.");
        return;
      }

      try {
        const response = await axios.get(`${API}/payments/checkout/status/${sessionId}`);
        
        if (response.data.payment_status === 'paid') {
          setError("");
          // Refresh user data to get updated subscription
          const userResponse = await axios.get(`${API}/auth/profile`);
          setCurrentUser(userResponse.data.user);
          localStorage.setItem('user', JSON.stringify(userResponse.data.user));
          setActiveTab("dashboard");
          return;
        } else if (response.data.status === 'expired') {
          setError("Payment session expired. Please try again.");
          return;
        }

        // Continue polling if payment is still pending
        attempts++;
        setTimeout(poll, pollInterval);
      } catch (error) {
        console.error("Error checking payment status:", error);
        setError("Error checking payment status. Please try again.");
      }
    };

    poll();
  };

  const loadUserWorkflows = async () => {
    try {
      if (!isAuthenticated) return;
      
      const response = await axios.get(`${API}/workflows`);
      setUserWorkflows(response.data.workflows);
    } catch (error) {
      console.error("Failed to load workflows:", error);
    }
  };

  const loadExecutiveDashboard = async () => {
    try {
      if (!isAuthenticated) return;
      
      const response = await axios.get(`${API}/dashboard/executive`);
      setExecutiveDashboard(response.data);
    } catch (error) {
      console.error("Failed to load executive dashboard:", error);
    }
  };

  // Check authentication on component mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userStr = localStorage.getItem('user');
    
    if (token && userStr) {
      try {
        const user = JSON.parse(userStr);
        setIsAuthenticated(true);
        setCurrentUser(user);
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      } catch (error) {
        console.error("Error parsing stored user data:", error);
        localStorage.removeItem('token');
        localStorage.removeItem('user');
      }
    }
  }, []);

  // Check for payment success on component mount
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session_id');
    
    if (sessionId) {
      checkPaymentStatus(sessionId);
      // Clean URL
      window.history.replaceState(null, null, window.location.pathname);
    }
  }, []);

  // Load data when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      loadSubscriptionPlans();
      loadUserWorkflows();
      if (currentUser?.subscription_tier === 'enterprise') {
        loadExecutiveDashboard();
      }
    }
  }, [isAuthenticated, currentUser]);

  // Phase 4: OAuth Handler Functions
  const handleGoogleLogin = async (credentialResponse) => {
    setLoadingState('login', true);
    setError("");

    try {
      const response = await axios.post(`${API}/auth/google/token`, {
        token: credentialResponse.credential
      });

      const { access_token, user } = response.data;
      
      // Store token and user info
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      
      // Set authentication state
      setIsAuthenticated(true);
      setCurrentUser(user);
      setShowLoginModal(false);
      
      // Configure axios default headers
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
    } catch (error) {
      console.error("Google login error:", error);
      setError(error.response?.data?.detail || "Google login failed. Please try again.");
    } finally {
      setLoadingState('login', false);
    }
  };

  const handleAppleLogin = async (response) => {
    setLoadingState('login', true);
    setError("");

    try {
      const authResponse = await axios.post(`${API}/auth/apple/token`, {
        code: response.authorization?.code,
        id_token: response.authorization?.id_token,
        name: response.user?.name
      });

      const { access_token, user } = authResponse.data;
      
      // Store token and user info
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      
      // Set authentication state
      setIsAuthenticated(true);
      setCurrentUser(user);
      setShowLoginModal(false);
      
      // Configure axios default headers
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
    } catch (error) {
      console.error("Apple login error:", error);
      setError(error.response?.data?.detail || "Apple login failed. Please try again.");
    } finally {
      setLoadingState('login', false);
    }
  };

  const resetForm = () => {
    setTherapyArea("");
    setProductName("");
    setAnalysis(null);
    setFunnel(null);
    setCompetitiveData(null);
    setScenarioModels(null);
    setClinicalTrials([]);
    setError("");
    setActiveTab("analysis");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h1 className="text-xl font-bold text-gray-900">Pharma Intelligence Platform</h1>
            </div>
            <div className="flex items-center space-x-3">
              {isAuthenticated ? (
                <>
                  <span className="text-sm text-gray-600">
                    Welcome, {currentUser?.first_name}
                  </span>
                  <div className={`px-2 py-1 text-xs rounded ${
                    currentUser?.subscription_tier === 'enterprise' ? 'bg-purple-100 text-purple-800' :
                    currentUser?.subscription_tier === 'professional' ? 'bg-blue-100 text-blue-800' :
                    currentUser?.subscription_tier === 'basic' ? 'bg-green-100 text-green-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {currentUser?.subscription_tier === 'free' ? 'Free' : currentUser?.subscription_tier?.charAt(0).toUpperCase() + currentUser?.subscription_tier?.slice(1)}
                  </div>
                  <button
                    onClick={() => setShowHistory(!showHistory)}
                    className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors flex items-center space-x-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>History</span>
                  </button>
                  <button
                    onClick={resetForm}
                    className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
                  >
                    New Analysis
                  </button>
                  <button
                    onClick={handleLogout}
                    className="px-4 py-2 text-sm text-red-600 hover:text-red-900 hover:bg-red-50 rounded-md transition-colors"
                  >
                    Logout
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => setShowLoginModal(true)}
                    className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
                  >
                    Login
                  </button>
                  <button
                    onClick={() => setShowRegisterModal(true)}
                    className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
                  >
                    Sign Up
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-8">
          <div className="mb-6">
            <img 
              src="https://images.unsplash.com/photo-1663363912772-b2f34992c805?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzV8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwYW5hbHlzaXN8ZW58MHx8fHwxNzUyOTIzMzQyfDA&ixlib=rb-4.1.0&q=85"
              alt="Medical Analysis"
              className="mx-auto h-32 w-auto rounded-lg shadow-lg"
            />
          </div>
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Comprehensive Pharma Intelligence Platform
          </h2>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto">
            AI-powered therapy area analysis, competitive intelligence, scenario modeling, 
            clinical trial research, and advanced forecasting for pharmaceutical professionals.
          </p>
        </div>

        {/* History Sidebar */}
        {showHistory && (
          <div className="fixed inset-y-0 right-0 w-80 bg-white shadow-xl z-50 overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Analysis History</h3>
                <button
                  onClick={() => setShowHistory(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="space-y-3">
                {savedAnalyses.map((savedAnalysis) => (
                  <div
                    key={savedAnalysis.id}
                    onClick={() => loadAnalysis(savedAnalysis.id)}
                    className="p-3 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors"
                  >
                    <div className="font-medium text-sm">{savedAnalysis.therapy_area}</div>
                    {savedAnalysis.product_name && (
                      <div className="text-xs text-blue-600">{savedAnalysis.product_name}</div>
                    )}
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(savedAnalysis.created_at).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6 sticky top-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h3>
              
              {/* Gemini API Key Input (Optional) */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gemini API Key (Optional)
                </label>
                <input
                  type="password"
                  value={geminiKey}
                  onChange={(e) => setGeminiKey(e.target.value)}
                  placeholder="AI..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  For multi-model ensemble analysis
                </p>
              </div>

              {/* Perplexity API Key Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Perplexity API Key (Optional)
                </label>
                <input
                  type="password"
                  value={perplexityKey}
                  onChange={(e) => setPerplexityKey(e.target.value)}
                  placeholder="pplx-..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  For enhanced real-time search capabilities
                </p>
              </div>

              {/* API Key Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Anthropic API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-ant-api03-..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Get your key from <a href="https://console.anthropic.com/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">console.anthropic.com</a>
                </p>
              </div>

              {/* Therapy Area Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Therapy Area *
                </label>
                <input
                  type="text"
                  value={therapyArea}
                  onChange={(e) => setTherapyArea(e.target.value)}
                  placeholder="e.g., Oncology - Lung Cancer, Cardiology, Diabetes"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Real-time Search Input */}
              {perplexityKey && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Real-time Intelligence Search
                  </label>
                  <input
                    type="text"
                    value={realTimeSearch}
                    onChange={(e) => setRealTimeSearch(e.target.value)}
                    placeholder="e.g., Latest GIST market data, Qinlock competitive analysis"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  />
                  <button
                    onClick={handleRealTimeSearch}
                    disabled={loadingStates.search}
                    className={`w-full mt-2 px-4 py-2 rounded-md font-medium transition-colors ${
                      loadingStates.search
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {loadingStates.search ? 'Searching...' : '🔍 Real-time Search'}
                  </button>
                </div>
              )}

              {/* Product Name Input */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Product Name (Optional)
                </label>
                <input
                  type="text"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  placeholder="e.g., Keytruda, Humira"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Company Intelligence Button */}
              {perplexityKey && productName && (
                <div className="mb-4">
                  <button
                    onClick={handleCompanyIntelligence}
                    disabled={loadingStates.company}
                    className={`w-full px-4 py-2 rounded-md font-medium transition-colors ${
                      loadingStates.company
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-orange-600 hover:bg-orange-700 text-white'
                    }`}
                  >
                    {loadingStates.company ? 'Researching...' : '🏢 Company Intelligence'}
                  </button>
                  <p className="text-xs text-gray-500 mt-1">
                    Automated competitive research and investor intelligence
                  </p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={handleAnalyzeTherapy}
                  disabled={loadingStates.analysis}
                  className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                    loadingStates.analysis
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {loadingStates.analysis ? 'Analyzing...' : '🔬 Comprehensive Analysis'}
                </button>

                {analysis && (
                  <>
                    {/* Phase 2: Advanced Analytics Buttons */}
                    <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                      <h4 className="text-sm font-semibold text-purple-900 mb-3">🚀 Advanced Analytics (Phase 2)</h4>
                      <div className="grid grid-cols-2 gap-2 mb-3">
                        <button
                          onClick={handleCreateFinancialModel}
                          disabled={loadingStates.financial}
                          className={`px-3 py-2 text-xs rounded font-medium transition-colors ${
                            loadingStates.financial
                              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                              : 'bg-blue-600 hover:bg-blue-700 text-white'
                          }`}
                        >
                          {loadingStates.financial ? 'Creating...' : '💰 Financial Model'}
                        </button>

                        {perplexityKey && (
                          <button
                            onClick={handleCreateTimeline}
                            disabled={loadingStates.timeline}
                            className={`px-3 py-2 text-xs rounded font-medium transition-colors ${
                              loadingStates.timeline
                                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                : 'bg-green-600 hover:bg-green-700 text-white'
                            }`}
                          >
                            {loadingStates.timeline ? 'Creating...' : '📅 Timeline View'}
                          </button>
                        )}
                      </div>

                      <div className="grid grid-cols-2 gap-2 mb-3">
                        {perplexityKey && (
                          <button
                            onClick={() => handleCreateTemplate("therapy_specific")}
                            disabled={loadingStates.template}
                            className={`px-3 py-2 text-xs rounded font-medium transition-colors ${
                              loadingStates.template
                                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                : 'bg-orange-600 hover:bg-orange-700 text-white'
                            }`}
                          >
                            {loadingStates.template ? 'Creating...' : '📋 Custom Template'}
                          </button>
                        )}

                        <button
                          onClick={() => handleAdvancedVisualization("positioning_map", "competitive")}
                          disabled={loadingStates.visualization}
                          className={`px-3 py-2 text-xs rounded font-medium transition-colors ${
                            loadingStates.visualization
                              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                              : 'bg-purple-600 hover:bg-purple-700 text-white'
                          }`}
                        >
                          {loadingStates.visualization ? 'Creating...' : '🎯 Advanced Viz'}
                        </button>
                      </div>

                      <p className="text-xs text-purple-600">Advanced financial modeling, timeline views, custom templates & 2D visualizations</p>
                    </div>

                    {/* Phase 1: Core Analysis Buttons */}
                    <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <h4 className="text-sm font-semibold text-blue-900 mb-2">🔬 Core Analysis</h4>
                      
                      {perplexityKey && (
                        <button
                          onClick={handleEnsembleAnalysis}
                          disabled={loadingStates.ensemble}
                          className={`w-full px-4 py-3 rounded-md font-medium transition-colors mb-3 ${
                            loadingStates.ensemble
                              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                              : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white'
                          }`}
                        >
                          {loadingStates.ensemble ? 'Analyzing...' : '🤖 Multi-Model Ensemble Analysis'}
                        </button>
                      )}

                    <button
                      onClick={handleGenerateFunnel}
                      disabled={loadingStates.funnel}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        loadingStates.funnel
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-700 text-white'
                      }`}
                    >
                      {loadingStates.funnel ? 'Generating...' : '📊 Generate Forecast Funnel'}
                    </button>
                    </div>

                    <button
                      onClick={handleEnhancedCompetitive}
                      disabled={loadingStates.competitive}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        loadingStates.competitive
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-purple-600 hover:bg-purple-700 text-white'
                      }`}
                    >
                      {loadingStates.competitive ? 'Analyzing...' : '🏆 Enhanced Competitive Intel'}
                    </button>

                    <button
                      onClick={handleScenarioModeling}
                      disabled={loadingStates.scenarios}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        loadingStates.scenarios
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                      }`}
                    >
                      {loadingStates.scenarios ? 'Modeling...' : '🎯 Scenario Modeling'}
                    </button>

                    <button
                      onClick={handleSearchTrials}
                      disabled={loadingStates.trials}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        loadingStates.trials
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-teal-600 hover:bg-teal-700 text-white'
                      }`}
                    >
                      {loadingStates.trials ? 'Searching...' : '🔍 Clinical Trials Research'}
                    </button>
                  </>
                )}
              </div>

              {/* Phase 3: Real-World Evidence Integration Buttons */}
              {analysis && (
                <div className="mt-4 p-3 bg-green-50 rounded-lg border border-green-200">
                  <h4 className="text-sm font-semibold text-green-900 mb-2">🔬 Phase 3: Real-World Evidence</h4>
                  
                  <div className="space-y-2">
                    <button
                      onClick={handleRealWorldEvidence}
                      disabled={loadingStates.rwe}
                      className={`w-full px-3 py-2 text-xs rounded font-medium transition-colors ${
                        loadingStates.rwe
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-700 text-white'
                      }`}
                    >
                      {loadingStates.rwe ? 'Analyzing RWE...' : '📊 Generate Real-World Evidence'}
                    </button>
                    
                    <button
                      onClick={handleMarketAccessIntelligence}
                      disabled={loadingStates.market_access}
                      className={`w-full px-3 py-2 text-xs rounded font-medium transition-colors ${
                        loadingStates.market_access
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-teal-600 hover:bg-teal-700 text-white'
                      }`}
                    >
                      {loadingStates.market_access ? 'Analyzing Market Access...' : '🏥 Market Access Intelligence'}
                    </button>
                    
                    <button
                      onClick={handlePredictiveAnalytics}
                      disabled={loadingStates.predictive}
                      className={`w-full px-3 py-2 text-xs rounded font-medium transition-colors ${
                        loadingStates.predictive
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                      }`}
                    >
                      {loadingStates.predictive ? 'Running Predictive Models...' : '🔮 Predictive Analytics'}
                    </button>
                    
                    <button
                      onClick={handlePhase3Dashboard}
                      disabled={loadingStates.dashboard}
                      className={`w-full px-3 py-2 text-xs rounded font-medium transition-colors ${
                        loadingStates.dashboard
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-purple-600 hover:bg-purple-700 text-white'
                      }`}
                    >
                      {loadingStates.dashboard ? 'Loading Dashboard...' : '📈 Phase 3 Dashboard'}
                    </button>
                  </div>
                  
                  <p className="text-xs text-green-600 mt-2">Advanced RWE analysis, market access intelligence & ML-enhanced forecasting</p>
                </div>
              )}

              {/* Phase 4: Subscription & Account Management */}
              {isAuthenticated && (
                <div className="mt-4 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-200">
                  <h4 className="text-sm font-semibold text-indigo-900 mb-2">🎯 Account & Subscription</h4>
                  
                  <div className="space-y-2">
                    <div className="text-xs text-gray-600 mb-2">
                      Current Plan: <span className="font-medium text-indigo-700">
                        {currentUser?.subscription_tier === 'free' ? 'Free Plan' : 
                         currentUser?.subscription_tier?.charAt(0).toUpperCase() + currentUser?.subscription_tier?.slice(1)}
                      </span>
                    </div>
                    
                    {currentUser?.subscription_tier === 'free' && (
                      <button
                        onClick={() => setShowPaymentModal(true)}
                        className="w-full px-3 py-2 text-xs rounded font-medium bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white transition-colors"
                      >
                        ⭐ Upgrade to Premium
                      </button>
                    )}
                    
                    {currentUser?.subscription_tier === 'enterprise' && executiveDashboard && (
                      <button
                        onClick={() => setActiveTab("executive")}
                        className="w-full px-3 py-2 text-xs rounded font-medium bg-purple-600 hover:bg-purple-700 text-white transition-colors"
                      >
                        📊 Executive Dashboard
                      </button>
                    )}
                  </div>
                  
                  <p className="text-xs text-indigo-600 mt-2">Unlock advanced features with premium subscriptions</p>
                </div>
              )}

              {!isAuthenticated && (
                <div className="mt-4 p-3 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200">
                  <h4 className="text-sm font-semibold text-yellow-900 mb-2">🔐 Sign Up for More</h4>
                  <p className="text-xs text-yellow-700 mb-2">Create an account to access:</p>
                  <ul className="text-xs text-yellow-600 space-y-1 mb-3">
                    <li>• Save and organize analyses</li>
                    <li>• Access history and templates</li>
                    <li>• Premium features and workflows</li>
                    <li>• Automated reporting</li>
                  </ul>
                  
                  <div className="space-y-2">
                    <button
                      onClick={() => setShowRegisterModal(true)}
                      className="w-full px-3 py-2 text-xs rounded font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
                    >
                      🚀 Create Free Account
                    </button>
                    <button
                      onClick={() => setShowLoginModal(true)}
                      className="w-full px-3 py-2 text-xs rounded font-medium bg-gray-600 hover:bg-gray-700 text-white transition-colors"
                    >
                      👤 Login
                    </button>
                  </div>
                </div>
              )}

              {/* Export Options */}
              {analysis && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Export Analysis</h4>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleExport('pdf')}
                      disabled={exportLoading}
                      className="flex-1 px-3 py-2 text-xs bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors disabled:bg-gray-300"
                    >
                      📄 PDF Report
                    </button>
                    <button
                      onClick={() => handleExport('excel')}
                      disabled={exportLoading}
                      className="flex-1 px-3 py-2 text-xs bg-emerald-600 hover:bg-emerald-700 text-white rounded-md transition-colors disabled:bg-gray-300"
                    >
                      📊 Excel Model
                    </button>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2">
            {(analysis || funnel || competitiveData || scenarioModels || clinicalTrials.length > 0 || perplexityResults || companyIntelligence || ensembleResult || financialModel || timeline || customTemplate || advancedViz || realWorldEvidence || marketAccessIntel || predictiveAnalytics || phase3Dashboard || executiveDashboard) && (
              <div className="bg-white rounded-xl shadow-lg">
                {/* Advanced Tabs */}
                <div className="border-b border-gray-200">
                  <nav className="-mb-px flex flex-wrap">
                    <button
                      onClick={() => setActiveTab("analysis")}
                      className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                        activeTab === "analysis"
                          ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                          : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      🔬 Therapy Analysis
                    </button>
                    {ensembleResult && (
                      <button
                        onClick={() => setActiveTab("ensemble")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "ensemble"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🤖 Multi-Model Analysis
                      </button>
                    )}
                    {funnel && (
                      <button
                        onClick={() => setActiveTab("funnel")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "funnel"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        📊 Patient Flow Funnel
                      </button>
                    )}
                    {competitiveData && (
                      <button
                        onClick={() => setActiveTab("competitive")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "competitive"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🏆 Competitive Intel
                      </button>
                    )}
                    {scenarioModels && (
                      <button
                        onClick={() => setActiveTab("scenarios")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "scenarios"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🎯 Scenarios
                      </button>
                    )}
                    {financialModel && (
                      <button
                        onClick={() => setActiveTab("financial")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "financial"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        💰 Financial Model
                      </button>
                    )}
                    {timeline && (
                      <button
                        onClick={() => setActiveTab("timeline")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "timeline"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        📅 Timeline
                      </button>
                    )}
                    {customTemplate && (
                      <button
                        onClick={() => setActiveTab("template")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "template"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        📋 Template
                      </button>
                    )}
                    {advancedViz && (
                      <button
                        onClick={() => setActiveTab("visualization")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "visualization"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🎯 Advanced Viz
                      </button>
                    )}
                    {companyIntelligence && (
                      <button
                        onClick={() => setActiveTab("company")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "company"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🏢 Company Intel
                      </button>
                    )}
                    {perplexityResults && (
                      <button
                        onClick={() => setActiveTab("search")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "search"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🔍 Real-time Search
                      </button>
                    )}
                    {clinicalTrials.length > 0 && (
                      <button
                        onClick={() => setActiveTab("trials")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "trials"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🔍 Clinical Trials ({clinicalTrials.length})
                      </button>
                    )}
                    
                    {/* Phase 3: Real-World Evidence Integration Tabs */}
                    {realWorldEvidence && (
                      <button
                        onClick={() => setActiveTab("rwe")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "rwe"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        📊 Real-World Evidence
                      </button>
                    )}
                    {marketAccessIntel && (
                      <button
                        onClick={() => setActiveTab("market_access")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "market_access"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🏥 Market Access
                      </button>
                    )}
                    {predictiveAnalytics && (
                      <button
                        onClick={() => setActiveTab("predictive")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "predictive"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        🔮 Predictive Analytics
                      </button>
                    )}
                    {phase3Dashboard && (
                      <button
                        onClick={() => setActiveTab("dashboard")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "dashboard"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        📈 Phase 3 Dashboard
                      </button>
                    )}
                    
                    {/* Phase 4: Executive Dashboard Tab */}
                    {isAuthenticated && executiveDashboard && (
                      <button
                        onClick={() => setActiveTab("executive")}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap ${
                          activeTab === "executive"
                            ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        👑 Executive Dashboard
                      </button>
                    )}
                  </nav>
                </div>

                <div className="p-6">
                  {/* Financial Model Tab */}
                  {activeTab === "financial" && financialModel && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">💰 Advanced Financial Model</h3>
                        <p className="text-gray-600">
                          NPV, IRR, Monte Carlo analysis for {financialModel.therapy_area}
                          {financialModel.product_name && <span className="font-medium"> - {financialModel.product_name}</span>}
                        </p>
                      </div>

                      {/* Key Metrics */}
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <div className="text-sm text-gray-600">NPV</div>
                          <div className="text-2xl font-bold text-green-600">
                            ${financialModel.npv_analysis?.npv?.toFixed(0) || 0}M
                          </div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                          <div className="text-sm text-gray-600">IRR</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {financialModel.irr_analysis?.irr?.toFixed(1) || 0}%
                          </div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                          <div className="text-sm text-gray-600">Success Probability</div>
                          <div className="text-2xl font-bold text-purple-600">
                            {financialModel.monte_carlo_results?.risk_metrics?.probability_of_success?.toFixed(0) || 0}%
                          </div>
                        </div>
                        <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <div className="text-sm text-gray-600">Payback Period</div>
                          <div className="text-2xl font-bold text-orange-600">
                            {financialModel.npv_analysis?.payback_period || 'N/A'} years
                          </div>
                        </div>
                      </div>

                      {/* Financial Projections Chart */}
                      {financialModel.visualization_data?.cash_flow_chart && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">📈 Cash Flow Projections</h4>
                          <PlotlyChart 
                            data={financialModel.visualization_data.cash_flow_chart} 
                            id={`cashflow-chart-${financialModel.id}`}
                          />
                        </div>
                      )}

                      {/* Monte Carlo Results */}
                      {financialModel.visualization_data?.monte_carlo_chart && (
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">🎲 Monte Carlo Simulation</h4>
                          <PlotlyChart 
                            data={financialModel.visualization_data.monte_carlo_chart} 
                            id={`montecarlo-chart-${financialModel.id}`}
                          />
                        </div>
                      )}

                      {/* Sensitivity Analysis */}
                      {financialModel.visualization_data?.npv_sensitivity_chart && (
                        <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">🌪️ Sensitivity Analysis</h4>
                          <PlotlyChart 
                            data={financialModel.visualization_data.npv_sensitivity_chart} 
                            id={`sensitivity-chart-${financialModel.id}`}
                          />
                        </div>
                      )}

                      {/* Risk Metrics */}
                      <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">⚠️ Risk Assessment</h4>
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <div className="text-sm font-medium text-gray-900 mb-2">Key Risk Factors:</div>
                            {financialModel.risk_metrics?.key_risk_factors?.map((factor, index) => (
                              <div key={index} className="text-sm text-gray-700">• {factor}</div>
                            ))}
                          </div>
                          <div>
                            <div className="text-sm font-medium text-gray-900 mb-2">Risk Level:</div>
                            <div className={`inline-block px-3 py-1 rounded text-sm ${
                              financialModel.risk_metrics?.npv_risk === 'High' ? 'bg-red-100 text-red-800' :
                              financialModel.risk_metrics?.npv_risk === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-green-100 text-green-800'
                            }`}>
                              {financialModel.risk_metrics?.npv_risk || 'Medium'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Timeline Tab */}
                  {activeTab === "timeline" && timeline && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">📅 Interactive Timeline</h3>
                        <p className="text-gray-600">
                          Development and competitive milestones for {timeline.therapy_area}
                          {timeline.product_name && <span className="font-medium"> - {timeline.product_name}</span>}
                        </p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded">
                            📋 {timeline.milestones?.length || 0} Milestones
                          </span>
                          <span className="px-2 py-1 bg-purple-100 text-purple-800 text-sm rounded">
                            🏆 {timeline.competitive_milestones?.length || 0} Competitive Events
                          </span>
                        </div>
                      </div>

                      {/* Timeline Visualization */}
                      {timeline.visualization_data?.timeline_chart && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">📊 Timeline Visualization</h4>
                          <PlotlyChart 
                            data={timeline.visualization_data.timeline_chart} 
                            id={`timeline-chart-${timeline.id}`}
                          />
                        </div>
                      )}

                      {/* Regulatory Timeline */}
                      {timeline.regulatory_timeline?.timeline && (
                        <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">🏛️ Regulatory Timeline</h4>
                          <div className="space-y-2">
                            {timeline.regulatory_timeline.timeline.map((phase, index) => (
                              <div key={index} className="flex justify-between items-center p-2 bg-white rounded">
                                <div>
                                  <div className="font-medium text-sm">{phase.phase}</div>
                                  <div className="text-xs text-gray-600">{phase.description}</div>
                                </div>
                                <div className="text-right">
                                  <div className="text-sm font-medium">{phase.start_date}</div>
                                  {phase.duration_months > 0 && (
                                    <div className="text-xs text-gray-500">{phase.duration_months} months</div>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Key Milestones */}
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📋 Key Milestones</h4>
                          <div className="space-y-2">
                            {timeline.milestones?.slice(0, 5).map((milestone, index) => (
                              <div key={index} className="bg-white rounded p-2">
                                <div className="flex justify-between items-start">
                                  <div className="text-sm font-medium">{milestone.title}</div>
                                  <div className="text-xs text-gray-500">{milestone.date}</div>
                                </div>
                                <div className={`text-xs mt-1 px-2 py-1 rounded inline-block ${
                                  milestone.type === 'clinical' ? 'bg-blue-100 text-blue-800' :
                                  milestone.type === 'regulatory' ? 'bg-red-100 text-red-800' :
                                  milestone.type === 'commercial' ? 'bg-green-100 text-green-800' :
                                  'bg-gray-100 text-gray-800'
                                }`}>
                                  {milestone.type}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🏆 Competitive Events</h4>
                          <div className="space-y-2">
                            {timeline.competitive_milestones?.slice(0, 5).map((milestone, index) => (
                              <div key={index} className="bg-white rounded p-2">
                                <div className="flex justify-between items-start">
                                  <div className="text-sm font-medium">{milestone.title}</div>
                                  <div className="text-xs text-gray-500">{milestone.date}</div>
                                </div>
                                <div className="text-xs text-purple-600 mt-1">Competitive</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Custom Template Tab */}
                  {activeTab === "template" && customTemplate && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">📋 Custom Analysis Template</h3>
                        <p className="text-gray-600">
                          {customTemplate.template_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} template
                          {customTemplate.therapy_area && <span className="font-medium"> for {customTemplate.therapy_area}</span>}
                        </p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-1 bg-orange-100 text-orange-800 text-sm rounded">
                            📄 {customTemplate.sections?.length || 0} Sections
                          </span>
                          <span className="px-2 py-1 bg-green-100 text-green-800 text-sm rounded">
                            🌍 {customTemplate.region}
                          </span>
                        </div>
                      </div>

                      {/* Template Sections */}
                      <div className="space-y-4">
                        {customTemplate.sections?.map((section, index) => (
                          <div key={index} className={`rounded-lg p-4 border ${
                            section.type === 'clinical' ? 'bg-blue-50 border-blue-200' :
                            section.type === 'regulatory' ? 'bg-red-50 border-red-200' :
                            section.type === 'market' ? 'bg-green-50 border-green-200' :
                            section.type === 'financial' ? 'bg-yellow-50 border-yellow-200' :
                            'bg-gray-50 border-gray-200'
                          }`}>
                            <div className="flex justify-between items-start mb-2">
                              <h4 className="text-lg font-semibold text-gray-900">{section.title}</h4>
                              <div className="flex gap-2">
                                {section.required && (
                                  <span className="px-2 py-1 bg-red-100 text-red-600 text-xs rounded">Required</span>
                                )}
                                <span className={`px-2 py-1 text-xs rounded ${
                                  section.type === 'clinical' ? 'bg-blue-100 text-blue-800' :
                                  section.type === 'regulatory' ? 'bg-red-100 text-red-800' :
                                  section.type === 'market' ? 'bg-green-100 text-green-800' :
                                  section.type === 'financial' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-gray-100 text-gray-800'
                                }`}>
                                  {section.type}
                                </span>
                              </div>
                            </div>
                            <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                              {section.content}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Template Metadata */}
                      <div className="bg-gray-50 rounded-lg p-4 border">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">📊 Template Information</h4>
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <div className="text-sm text-gray-600">Word Count</div>
                            <div className="font-medium">{customTemplate.template_data?.word_count || 0}</div>
                          </div>
                          <div>
                            <div className="text-sm text-gray-600">Generated By</div>
                            <div className="font-medium">{customTemplate.template_data?.generated_by || 'AI'}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Advanced Visualization Tab */}
                  {activeTab === "visualization" && advancedViz && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🎯 Advanced Visualization</h3>
                        <p className="text-gray-600">
                          {advancedViz.visualization_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} 
                          visualization from {advancedViz.data_source} data
                        </p>
                      </div>

                      {/* Visualization Display */}
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6 border border-purple-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-4">📈 Interactive Visualization</h4>
                        {advancedViz.chart_data && !advancedViz.chart_data.includes('"error"') ? (
                          <PlotlyChart 
                            data={advancedViz.chart_data} 
                            id={`advanced-viz-${Date.now()}`}
                          />
                        ) : (
                          <div className="text-center py-8 text-gray-600">
                            <div className="text-4xl mb-2">📊</div>
                            <div>Visualization not available</div>
                            <div className="text-sm mt-1">
                              {advancedViz.chart_data?.includes('"error"') ? 
                                'Error generating visualization' : 
                                'Chart data processing...'
                              }
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Visualization Info */}
                      <div className="bg-gray-50 rounded-lg p-4 border">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">ℹ️ Visualization Details</h4>
                        <div className="grid md:grid-cols-3 gap-4">
                          <div>
                            <div className="text-sm text-gray-600">Type</div>
                            <div className="font-medium">{advancedViz.visualization_type}</div>
                          </div>
                          <div>
                            <div className="text-sm text-gray-600">Data Source</div>
                            <div className="font-medium">{advancedViz.data_source}</div>
                          </div>
                          <div>
                            <div className="text-sm text-gray-600">Generated</div>
                            <div className="font-medium">
                              {new Date(advancedViz.generated_at).toLocaleString()}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Multi-Model Ensemble Analysis Tab */}
                  {activeTab === "ensemble" && ensembleResult && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">
                          🤖 Multi-Model Ensemble Analysis
                        </h3>
                        <p className="text-gray-600 mb-3">
                          Advanced AI analysis combining insights from multiple models for {ensembleResult.therapy_area}
                          {ensembleResult.product_name && <span className="font-medium"> - {ensembleResult.product_name}</span>}
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(ensembleResult.confidence_scores).map(([model, score]) => (
                            <span key={model} className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                              🤖 {model}: {(score * 100).toFixed(0)}% confidence
                            </span>
                          ))}
                          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                            📊 Agreement: {(ensembleResult.model_agreement_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      {/* Model Agreement & Recommendation */}
                      <div className={`rounded-lg p-4 border-2 ${
                        ensembleResult.model_agreement_score > 0.8 ? 'bg-green-50 border-green-200' :
                        ensembleResult.model_agreement_score > 0.6 ? 'bg-yellow-50 border-yellow-200' :
                        'bg-red-50 border-red-200'
                      }`}>
                        <h4 className="text-lg font-semibold text-gray-900 mb-2">
                          {ensembleResult.model_agreement_score > 0.8 ? '✅' : 
                           ensembleResult.model_agreement_score > 0.6 ? '⚠️' : '🚨'} 
                          Ensemble Recommendation
                        </h4>
                        <p className="text-gray-700 font-medium mb-2">{ensembleResult.recommendation}</p>
                        <div className="text-sm text-gray-600">
                          Model Agreement Score: {(ensembleResult.model_agreement_score * 100).toFixed(1)}%
                        </div>
                      </div>

                      {/* Consensus Insights */}
                      {ensembleResult.consensus_insights.length > 0 && (
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🎯 Consensus Insights</h4>
                          <ul className="space-y-2">
                            {ensembleResult.consensus_insights.map((insight, index) => (
                              <li key={index} className="flex items-start">
                                <span className="text-blue-600 mr-2">•</span>
                                <span className="text-gray-700">{insight}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Conflicting Points */}
                      {ensembleResult.conflicting_points.length > 0 && (
                        <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">⚠️ Areas of Disagreement</h4>
                          <ul className="space-y-2">
                            {ensembleResult.conflicting_points.map((conflict, index) => (
                              <li key={index} className="flex items-start">
                                <span className="text-orange-600 mr-2">•</span>
                                <span className="text-gray-700">{conflict}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Individual Model Analyses */}
                      <div className="grid lg:grid-cols-3 gap-4">
                        {/* Claude Analysis */}
                        <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                          <h5 className="font-semibold text-indigo-900 mb-2 flex items-center">
                            🧠 Claude Analysis
                            <span className="ml-2 text-sm bg-indigo-200 px-2 py-1 rounded">
                              {(ensembleResult.claude_analysis.confidence_score * 100).toFixed(0)}%
                            </span>
                          </h5>
                          <div className="text-sm text-gray-700 max-h-40 overflow-y-auto">
                            {ensembleResult.claude_analysis.analysis.slice(0, 500)}...
                          </div>
                          {ensembleResult.claude_analysis.error && (
                            <div className="text-red-600 text-xs mt-2">
                              Error: {ensembleResult.claude_analysis.error}
                            </div>
                          )}
                        </div>

                        {/* Perplexity Intelligence */}
                        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                          <h5 className="font-semibold text-green-900 mb-2 flex items-center">
                            🔍 Perplexity Intelligence
                            <span className="ml-2 text-sm bg-green-200 px-2 py-1 rounded">
                              {(ensembleResult.perplexity_intelligence.confidence_score * 100).toFixed(0)}%
                            </span>
                          </h5>
                          <div className="text-sm text-gray-700 max-h-40 overflow-y-auto mb-2">
                            {ensembleResult.perplexity_intelligence.analysis.slice(0, 500)}...
                          </div>
                          {ensembleResult.perplexity_intelligence.citation_count > 0 && (
                            <div className="text-xs text-green-600">
                              📚 {ensembleResult.perplexity_intelligence.citation_count} sources cited
                            </div>
                          )}
                          {ensembleResult.perplexity_intelligence.error && (
                            <div className="text-red-600 text-xs mt-2">
                              Error: {ensembleResult.perplexity_intelligence.error}
                            </div>
                          )}
                        </div>

                        {/* Gemini Analysis (if available) */}
                        {ensembleResult.gemini_analysis && (
                          <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                            <h5 className="font-semibold text-purple-900 mb-2 flex items-center">
                              ✨ Gemini Analysis
                              <span className="ml-2 text-sm bg-purple-200 px-2 py-1 rounded">
                                {(ensembleResult.gemini_analysis.confidence_score * 100).toFixed(0)}%
                              </span>
                            </h5>
                            <div className="text-sm text-gray-700 max-h-40 overflow-y-auto">
                              {ensembleResult.gemini_analysis.analysis.slice(0, 500)}...
                            </div>
                            {ensembleResult.gemini_analysis.error && (
                              <div className="text-red-600 text-xs mt-2">
                                Error: {ensembleResult.gemini_analysis.error}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Synthesized Analysis */}
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6 border border-purple-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">🎭 Synthesized Intelligence</h4>
                        <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                          {ensembleResult.ensemble_synthesis}
                        </div>
                      </div>

                      {/* Sources */}
                      {ensembleResult.sources.length > 0 && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📚 Intelligence Sources</h4>
                          <div className="grid md:grid-cols-3 gap-2">
                            {ensembleResult.sources.slice(0, 9).map((source, index) => (
                              <a
                                key={index}
                                href={source}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs bg-white px-2 py-1 rounded border text-blue-600 hover:bg-blue-100 truncate"
                              >
                                🔗 Source {index + 1}
                              </a>
                            ))}
                          </div>
                          {ensembleResult.sources.length > 9 && (
                            <div className="text-xs text-gray-500 mt-2">
                              ... and {ensembleResult.sources.length - 9} more sources
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Therapy Analysis Tab */}
                  {activeTab === "analysis" && analysis && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">
                          {analysis.therapy_area}
                          {analysis.product_name && (
                            <span className="text-lg text-blue-600"> - {analysis.product_name}</span>
                          )}
                        </h3>
                        <div className="flex flex-wrap gap-2 mt-3">
                          {analysis.risk_assessment && (
                            <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm rounded-full">
                              Risk Score: {analysis.risk_assessment.overall_score}/10
                            </span>
                          )}
                          {analysis.clinical_trials_data && (
                            <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
                              {analysis.clinical_trials_data.length} Active Trials
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Enhanced Analysis Sections */}
                      {[
                        { title: "🦠 Disease Summary", content: analysis.disease_summary, color: "bg-blue-50" },
                        { title: "📋 Staging", content: analysis.staging, color: "bg-green-50" },
                        { title: "🧬 Biomarkers", content: analysis.biomarkers, color: "bg-purple-50" },
                        { title: "🔄 Treatment Algorithm", content: analysis.treatment_algorithm, color: "bg-orange-50" },
                        { title: "🚶 Patient Journey", content: analysis.patient_journey, color: "bg-teal-50" }
                      ].map((section, index) => (
                        <div key={index} className={`${section.color} rounded-lg p-4 border`}>
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">{section.title}</h4>
                          <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{section.content}</div>
                        </div>
                      ))}

                      {/* Risk Assessment Summary */}
                      {analysis.risk_assessment && (
                        <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">⚠️ Risk Assessment</h4>
                          <div className="grid md:grid-cols-2 gap-4">
                            {Object.entries(analysis.risk_assessment).map(([key, value]) => {
                              if (key === 'overall_score' || key === 'full_assessment') return null;
                              return (
                                <div key={key} className="bg-white rounded-md p-3">
                                  <div className="font-medium text-sm text-gray-900 mb-1">
                                    {key.replace('_', ' ').toUpperCase()}
                                  </div>
                                  <div className={`text-xs px-2 py-1 rounded ${
                                    value.level === 'High' ? 'bg-red-100 text-red-800' :
                                    value.level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-green-100 text-green-800'
                                  }`}>
                                    {value.level} Risk
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Enhanced Funnel Tab */}
                  {activeTab === "funnel" && funnel && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">Patient Flow Funnel & Forecasting</h3>
                        <p className="text-gray-600">Advanced forecasting model for {funnel.therapy_area}</p>
                      </div>

                      {/* Interactive Funnel Visualization */}
                      {funnel.visualization_data && funnel.visualization_data.funnel_chart && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">📊 Interactive Patient Flow Funnel</h4>
                          <PlotlyChart 
                            data={funnel.visualization_data.funnel_chart} 
                            id={`funnel-chart-${funnel.id}`}
                          />
                        </div>
                      )}

                      {/* Funnel Stages */}
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold text-gray-900 mb-4">🔄 Funnel Stages</h4>
                        <div className="space-y-3">
                          {funnel.funnel_stages.map((stage, index) => (
                            <div key={index} className="bg-white rounded-md p-4 border-l-4 border-blue-500 shadow-sm">
                              <div className="flex justify-between items-start mb-2">
                                <h5 className="font-medium text-gray-900">{stage.stage}</h5>
                                <span className="text-lg font-bold text-blue-600">{stage.percentage}</span>
                              </div>
                              <p className="text-sm text-gray-600 mb-2">{stage.description}</p>
                              <p className="text-xs text-gray-500 bg-gray-100 p-2 rounded">{stage.notes}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* TAM Analysis */}
                      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">💰 Total Addressable Market</h4>
                        <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{funnel.total_addressable_population}</div>
                      </div>

                      {/* Scenario Models Visualization */}
                      {funnel.visualization_data && funnel.visualization_data.scenario_chart && (
                        <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">📈 Market Forecast Scenarios</h4>
                          <PlotlyChart 
                            data={funnel.visualization_data.scenario_chart} 
                            id={`scenario-chart-${funnel.id}`}
                          />
                        </div>
                      )}

                      {/* Forecasting Notes */}
                      <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">📝 Forecasting Methodology</h4>
                        <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{funnel.forecasting_notes}</div>
                      </div>
                    </div>
                  )}

                  {/* Enhanced Competitive Intelligence Tab */}
                  {activeTab === "competitive" && competitiveData && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🏆 Enhanced Competitive Intelligence</h3>
                        <p className="text-gray-600">Real-time competitive analysis for {therapyArea}</p>
                        {competitiveData.analysis_type && (
                          <div className="mt-2 text-sm text-blue-600 font-medium">
                            🤖 {competitiveData.analysis_type} | 🔗 {competitiveData.total_sources} Sources
                          </div>
                        )}
                      </div>

                      {/* Real-time Intelligence Section */}
                      {competitiveData.real_time_intelligence && (
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🔍 Real-time Market Intelligence</h4>
                          <div className="text-gray-700 leading-relaxed whitespace-pre-wrap mb-4">
                            {competitiveData.real_time_intelligence.content}
                          </div>
                          
                          {competitiveData.real_time_intelligence.sources && competitiveData.real_time_intelligence.sources.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-blue-200">
                              <div className="text-sm font-medium text-gray-900 mb-2">📚 Live Sources:</div>
                              <div className="flex flex-wrap gap-2">
                                {competitiveData.real_time_intelligence.sources.slice(0, 5).map((source, index) => (
                                  <a 
                                    key={index}
                                    href={source} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-xs bg-white px-2 py-1 rounded border text-blue-600 hover:bg-blue-100"
                                  >
                                    Source {index + 1}
                                  </a>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Enhanced Analysis Section */}
                      {competitiveData.enhanced_analysis && (
                        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🤖 AI-Enhanced Analysis</h4>
                          <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                            {competitiveData.enhanced_analysis}
                          </div>
                        </div>
                      )}

                      {/* Legacy competitive data structure support */}
                      {competitiveData.competitors && (
                        <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">🎯 Key Market Players</h4>
                          <div className="grid md:grid-cols-2 gap-4">
                            {competitiveData.competitors.slice(0, 6).map((competitor, index) => (
                              <div key={index} className="bg-white rounded-md p-4 shadow-sm border">
                                <div className="flex justify-between items-start mb-2">
                                  <h5 className="font-medium text-gray-900">{competitor.name}</h5>
                                  <span className="text-sm font-semibold text-purple-600">
                                    {competitor.market_share}% market
                                  </span>
                                </div>
                                {competitor.products && (
                                  <p className="text-sm text-blue-600 mb-1">Products: {Array.isArray(competitor.products) ? competitor.products.join(', ') : competitor.products}</p>
                                )}
                                <p className="text-xs text-green-600 mb-1">✓ {competitor.strengths}</p>
                                {competitor.weaknesses && (
                                  <p className="text-xs text-red-600">⚠ {competitor.weaknesses}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Combined Insights */}
                      {competitiveData.combined_insights && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">💡 Complete Intelligence Report</h4>
                          <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto">
                            {competitiveData.combined_insights}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Scenario Modeling Tab - Enhanced */}
                  {activeTab === "scenarios" && scenarioModels && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🎯 Market Forecasting Scenarios</h3>
                        <p className="text-gray-600">
                          Multi-scenario analysis for {therapyArea}
                          {analysis?.product_name && <span className="font-medium"> - {analysis.product_name}</span>}
                        </p>
                      </div>

                      {/* Scenario Summary Cards */}
                      <div className="grid lg:grid-cols-3 gap-4 mb-6">
                        {Object.entries(scenarioModels).map(([scenario, data]) => (
                          <div key={scenario} className={`rounded-lg p-4 border-2 ${
                            scenario === 'optimistic' ? 'bg-green-50 border-green-200' :
                            scenario === 'pessimistic' ? 'bg-red-50 border-red-200' :
                            'bg-blue-50 border-blue-200'
                          }`}>
                            <h4 className="text-lg font-bold mb-3 capitalize flex items-center">
                              {scenario === 'optimistic' ? '📈' : scenario === 'pessimistic' ? '📉' : '📊'} 
                              <span className="ml-2">{scenario} Case</span>
                            </h4>
                            
                            {/* Key Metrics */}
                            <div className="space-y-3">
                              {data.peak_sales && (
                                <div className="bg-white rounded-md p-3">
                                  <div className="text-sm text-gray-600">Peak Annual Sales</div>
                                  <div className="text-xl font-bold text-gray-900">${data.peak_sales}M</div>
                                </div>
                              )}

                              {data.projections && data.projections.length > 0 && (
                                <div className="bg-white rounded-md p-3">
                                  <div className="text-sm text-gray-600 mb-2">6-Year Revenue Trajectory</div>
                                  <div className="space-y-1">
                                    {data.projections.slice(0, 6).map((projection, index) => (
                                      <div key={index} className="flex justify-between text-sm">
                                        <span className="text-gray-600">{2024 + index}</span>
                                        <span className="font-medium">${projection}M</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* Market Share */}
                              {data.market_share_trajectory && (
                                <div className="bg-white rounded-md p-3">
                                  <div className="text-sm text-gray-600 mb-1">Peak Market Share</div>
                                  <div className="text-lg font-bold text-blue-600">
                                    {Math.max(...data.market_share_trajectory)}%
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Key Assumptions & Success Factors */}
                      <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🎯 Key Assumptions</h4>
                          <div className="space-y-3">
                            {Object.entries(scenarioModels).map(([scenario, data]) => (
                              <div key={scenario} className="bg-white rounded-md p-3">
                                <div className="font-medium text-sm text-gray-900 capitalize mb-2">{scenario} Scenario</div>
                                <ul className="text-sm text-gray-600 space-y-1">
                                  {data.assumptions && data.assumptions.slice(0, 3).map((assumption, index) => (
                                    <li key={index} className="flex items-start">
                                      <span className="mr-2">•</span>
                                      <span>{assumption}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🔑 Success Factors</h4>
                          <div className="space-y-3">
                            {Object.entries(scenarioModels).map(([scenario, data]) => (
                              <div key={scenario} className="bg-white rounded-md p-3">
                                <div className="font-medium text-sm text-gray-900 capitalize mb-2">{scenario} Keys</div>
                                <ul className="text-sm text-gray-600 space-y-1">
                                  {data.key_factors && data.key_factors.slice(0, 2).map((factor, index) => (
                                    <li key={index} className="flex items-start">
                                      <span className="mr-2">✓</span>
                                      <span>{factor}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Full Analysis Details */}
                      {Object.values(scenarioModels).some(data => data.full_analysis) && (
                        <div className="bg-gray-50 rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📋 Detailed Analysis</h4>
                          <div className="max-h-60 overflow-y-auto text-sm text-gray-700 leading-relaxed">
                            {Object.values(scenarioModels).find(data => data.full_analysis)?.full_analysis}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Company Intelligence Tab */}
                  {activeTab === "company" && companyIntelligence && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">
                          🏢 Company Intelligence: {companyIntelligence.product_name}
                        </h3>
                        <div className="flex flex-wrap gap-2 text-sm">
                          <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full">
                            🏛️ {companyIntelligence.parent_company}
                          </span>
                          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full">
                            💊 {companyIntelligence.market_class}
                          </span>
                          <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full">
                            🔗 {companyIntelligence.sources_scraped.length} Sources
                          </span>
                        </div>
                      </div>

                      {/* Company Overview */}
                      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">🏛️ Company Overview</h4>
                        <div className="grid md:grid-cols-2 gap-4">
                          <div className="bg-white rounded-md p-3">
                            <div className="text-sm text-gray-600">Parent Company</div>
                            <div className="font-medium text-gray-900">{companyIntelligence.parent_company}</div>
                          </div>
                          <div className="bg-white rounded-md p-3">
                            <div className="text-sm text-gray-600">Website</div>
                            <a 
                              href={companyIntelligence.company_website}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-medium text-blue-600 hover:underline"
                            >
                              {companyIntelligence.company_website}
                            </a>
                          </div>
                          <div className="bg-white rounded-md p-3">
                            <div className="text-sm text-gray-600">Drug Class</div>
                            <div className="font-medium text-gray-900">{companyIntelligence.market_class}</div>
                          </div>
                          <div className="bg-white rounded-md p-3">
                            <div className="text-sm text-gray-600">Analysis Date</div>
                            <div className="font-medium text-gray-900">
                              {new Date(companyIntelligence.timestamp).toLocaleDateString()}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Financial Highlights */}
                      {companyIntelligence.financial_metrics && companyIntelligence.financial_metrics.highlights && companyIntelligence.financial_metrics.highlights.length > 0 && (
                        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">💰 Financial Highlights</h4>
                          <div className="grid md:grid-cols-3 gap-3">
                            {companyIntelligence.financial_metrics.highlights.slice(0, 6).map((highlight, index) => (
                              <div key={index} className="bg-white rounded-md p-3 text-center">
                                <div className="font-bold text-green-600 text-lg">{highlight.metric}</div>
                                <div className="text-xs text-gray-500">{highlight.source}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Press Releases & Investor Data */}
                      {companyIntelligence.press_releases && companyIntelligence.press_releases.length > 0 && (
                        <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📰 Recent Press Releases</h4>
                          <div className="space-y-2">
                            {companyIntelligence.press_releases.slice(0, 5).map((release, index) => (
                              <div key={index} className="bg-white rounded-md p-3 border-l-4 border-orange-500">
                                <a 
                                  href={release.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-sm font-medium text-blue-600 hover:underline"
                                >
                                  {release.title}
                                </a>
                                <div className="text-xs text-gray-500 mt-1">Press Release</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Competitive Products */}
                      {companyIntelligence.competitive_products && companyIntelligence.competitive_products.length > 0 && (
                        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🏆 Competitive Landscape</h4>
                          <div className="grid md:grid-cols-2 gap-3">
                            {companyIntelligence.competitive_products.slice(0, 6).map((product, index) => (
                              <div key={index} className="bg-white rounded-md p-3">
                                <div className="font-medium text-gray-900 mb-1">{product.name}</div>
                                <div className="text-sm text-blue-600 mb-1">{product.company}</div>
                                <div className="text-xs text-gray-600">{product.description?.slice(0, 100)}...</div>
                                {product.approval_status && (
                                  <div className="text-xs text-green-600 mt-1">{product.approval_status}</div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Recent Developments */}
                      {companyIntelligence.recent_developments && companyIntelligence.recent_developments.length > 0 && (
                        <div className="bg-teal-50 rounded-lg p-4 border border-teal-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📈 Recent Developments</h4>
                          <div className="space-y-2">
                            {companyIntelligence.recent_developments.slice(0, 5).map((dev, index) => (
                              <div key={index} className="bg-white rounded-md p-3 border-l-4 border-teal-500">
                                <div className="text-sm text-gray-700">{dev.update}</div>
                                <div className="text-xs text-gray-500 mt-1">
                                  {new Date(dev.timestamp).toLocaleDateString()}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Sources Accessed */}
                      <div className="bg-gray-50 rounded-lg p-4 border">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">📚 Intelligence Sources</h4>
                        <div className="flex flex-wrap gap-2">
                          {companyIntelligence.sources_scraped.map((source, index) => (
                            <a
                              key={index}
                              href={source}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs bg-white px-2 py-1 rounded border text-blue-600 hover:bg-blue-100"
                            >
                              Source {index + 1}
                            </a>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Real-time Search Tab */}
                  {activeTab === "search" && perplexityResults && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🔍 Real-time Intelligence Search</h3>
                        <p className="text-gray-600">Latest market intelligence with verified sources</p>
                        <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
                          <span>📊 Query: {perplexityResults.search_query}</span>
                          <span>📅 {new Date(perplexityResults.timestamp).toLocaleString()}</span>
                          <span>🔗 {perplexityResults.citations.length} Sources</span>
                        </div>
                      </div>

                      {/* Search Results */}
                      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-4">📋 Intelligence Report</h4>
                        <div className="text-gray-700 leading-relaxed whitespace-pre-wrap mb-4">
                          {perplexityResults.content}
                        </div>
                        
                        {/* Citations */}
                        {perplexityResults.citations.length > 0 && (
                          <div className="mt-6 pt-4 border-t border-green-200">
                            <h5 className="text-md font-semibold text-gray-900 mb-3">📚 Verified Sources:</h5>
                            <div className="grid md:grid-cols-2 gap-3">
                              {perplexityResults.citations.map((citation, index) => (
                                <div key={index} className="bg-white rounded-md p-3 shadow-sm border">
                                  <a 
                                    href={citation} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:text-blue-800 text-sm font-medium block"
                                  >
                                    📖 Source {index + 1}
                                  </a>
                                  <div className="text-xs text-gray-500 mt-1 truncate">
                                    {citation}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Clinical Trials Tab */}
                  {activeTab === "trials" && clinicalTrials.length > 0 && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🔍 Clinical Trials Research</h3>
                        <p className="text-gray-600">Found {clinicalTrials.length} relevant clinical trials for {therapyArea}</p>
                      </div>

                      <div className="space-y-4">
                        {clinicalTrials.slice(0, 10).map((trial, index) => {
                          const protocol = trial.protocolSection || {};
                          const identification = protocol.identificationModule || {};
                          const status = protocol.statusModule || {};
                          const design = protocol.designModule || {};
                          
                          return (
                            <div key={index} className="bg-gray-50 rounded-lg p-4 border">
                              <div className="flex justify-between items-start mb-2">
                                <div className="flex-1">
                                  <h4 className="font-semibold text-gray-900 mb-1">
                                    {identification.nctId || 'N/A'}
                                  </h4>
                                  <h5 className="text-sm text-gray-700 mb-2">
                                    {identification.briefTitle || identification.officialTitle || 'No title available'}
                                  </h5>
                                </div>
                                <div className="flex flex-col items-end space-y-1">
                                  {status.overallStatus && (
                                    <span className={`px-2 py-1 text-xs rounded-full ${
                                      status.overallStatus === 'Recruiting' ? 'bg-green-100 text-green-800' :
                                      status.overallStatus === 'Active, not recruiting' ? 'bg-blue-100 text-blue-800' :
                                      status.overallStatus === 'Completed' ? 'bg-gray-100 text-gray-800' :
                                      'bg-yellow-100 text-yellow-800'
                                    }`}>
                                      {status.overallStatus}
                                    </span>
                                  )}
                                  {design.phases && design.phases.length > 0 && (
                                    <span className="px-2 py-1 text-xs bg-purple-100 text-purple-800 rounded-full">
                                      {design.phases.join(', ')}
                                    </span>
                                  )}
                                </div>
                              </div>
                              
                              {protocol.conditionsModule && protocol.conditionsModule.conditions && (
                                <div className="mb-2">
                                  <span className="text-xs text-gray-500">Conditions: </span>
                                  <span className="text-xs text-blue-600">
                                    {protocol.conditionsModule.conditions.slice(0, 3).join(', ')}
                                  </span>
                                </div>
                              )}

                              {status.studyFirstSubmitDate && (
                                <div className="text-xs text-gray-500">
                                  First Submitted: {new Date(status.studyFirstSubmitDate).toLocaleDateString()}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Phase 3: Real-World Evidence Tab */}
                  {activeTab === "rwe" && realWorldEvidence && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">📊 Real-World Evidence Analysis</h3>
                        <p className="text-gray-600">
                          Comprehensive RWE analysis for {realWorldEvidence.therapy_area}
                          {realWorldEvidence.product_name && <span className="font-medium"> - {realWorldEvidence.product_name}</span>}
                        </p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-1 bg-green-100 text-green-800 text-sm rounded">
                            📈 Evidence Quality: {(realWorldEvidence.evidence_quality_score * 100).toFixed(0)}%
                          </span>
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded">
                            📋 Data Sources: {realWorldEvidence.data_sources?.length || 0}
                          </span>
                        </div>
                      </div>

                      {/* Key RWE Metrics */}
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <div className="text-sm text-gray-600">Effectiveness Score</div>
                          <div className="text-2xl font-bold text-green-600">
                            {realWorldEvidence.effectiveness_data?.effectiveness_score || "N/A"}
                          </div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                          <div className="text-sm text-gray-600">Safety Profile</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {realWorldEvidence.safety_profile?.overall_rating || "Good"}
                          </div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                          <div className="text-sm text-gray-600">Patient Satisfaction</div>
                          <div className="text-2xl font-bold text-purple-600">
                            {realWorldEvidence.patient_outcomes?.satisfaction_score || "High"}
                          </div>
                        </div>
                        <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <div className="text-sm text-gray-600">Cost-Effectiveness</div>
                          <div className="text-2xl font-bold text-orange-600">
                            {realWorldEvidence.cost_effectiveness?.cost_per_qaly || "Favorable"}
                          </div>
                        </div>
                      </div>

                      {/* RWE Analysis Sections */}
                      <div className="space-y-4">
                        {/* Effectiveness Data */}
                        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🎯 Effectiveness Data</h4>
                          <div className="text-sm text-gray-700">
                            {realWorldEvidence.effectiveness_data?.content || "Real-world effectiveness analysis shows promising outcomes across diverse patient populations."}
                          </div>
                          {realWorldEvidence.effectiveness_data?.key_points && (
                            <ul className="mt-2 space-y-1">
                              {realWorldEvidence.effectiveness_data.key_points.slice(0, 5).map((point, index) => (
                                <li key={index} className="text-sm text-gray-600">• {point}</li>
                              ))}
                            </ul>
                          )}
                        </div>

                        {/* Safety Profile */}
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🛡️ Safety Profile</h4>
                          <div className="text-sm text-gray-700">
                            {realWorldEvidence.safety_profile?.content || "Comprehensive safety monitoring in real-world settings demonstrates acceptable risk profile."}
                          </div>
                        </div>

                        {/* Comparative Effectiveness */}
                        {realWorldEvidence.comparative_effectiveness && realWorldEvidence.comparative_effectiveness.length > 0 && (
                          <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                            <h4 className="text-lg font-semibold text-gray-900 mb-3">⚖️ Comparative Effectiveness</h4>
                            <div className="space-y-2">
                              {realWorldEvidence.comparative_effectiveness.slice(0, 3).map((comparison, index) => (
                                <div key={index} className="bg-white p-3 rounded border">
                                  <div className="text-sm font-medium text-gray-900">{comparison.comparison}</div>
                                  <div className="text-sm text-gray-600">{comparison.outcome}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Recommendations */}
                        {realWorldEvidence.recommendations && realWorldEvidence.recommendations.length > 0 && (
                          <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                            <h4 className="text-lg font-semibold text-gray-900 mb-3">💡 Recommendations</h4>
                            <ul className="space-y-1">
                              {realWorldEvidence.recommendations.slice(0, 5).map((recommendation, index) => (
                                <li key={index} className="text-sm text-gray-700">• {recommendation}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Phase 3: Market Access Intelligence Tab */}
                  {activeTab === "market_access" && marketAccessIntel && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🏥 Market Access Intelligence</h3>
                        <p className="text-gray-600">
                          Comprehensive market access analysis for {marketAccessIntel.therapy_area}
                          {marketAccessIntel.product_name && <span className="font-medium"> - {marketAccessIntel.product_name}</span>}
                        </p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-1 bg-teal-100 text-teal-800 text-sm rounded">
                            📊 Readiness Score: {(marketAccessIntel.market_readiness_score * 100).toFixed(0)}%
                          </span>
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded">
                            🌍 Markets Analyzed: {marketAccessIntel.target_markets?.length || 0}
                          </span>
                        </div>
                      </div>

                      {/* Market Access Key Metrics */}
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-teal-50 p-4 rounded-lg border border-teal-200">
                          <div className="text-sm text-gray-600">Payer Coverage</div>
                          <div className="text-2xl font-bold text-teal-600">
                            {marketAccessIntel.payer_landscape?.coverage_rate || "Favorable"}
                          </div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                          <div className="text-sm text-gray-600">Access Timeline</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {marketAccessIntel.approval_timelines?.average_months || "12-18"} months
                          </div>
                        </div>
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <div className="text-sm text-gray-600">Pricing Position</div>
                          <div className="text-2xl font-bold text-green-600">
                            {marketAccessIntel.pricing_analysis?.position || "Competitive"}
                          </div>
                        </div>
                      </div>

                      {/* Market Access Analysis Sections */}
                      <div className="space-y-4">
                        {/* Payer Landscape */}
                        <div className="bg-teal-50 rounded-lg p-4 border border-teal-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">💼 Payer Landscape</h4>
                          <div className="text-sm text-gray-700">
                            {marketAccessIntel.payer_landscape?.content || "Analysis of key payers, decision-making processes, and reimbursement mechanisms."}
                          </div>
                        </div>

                        {/* Access Barriers */}
                        {marketAccessIntel.access_barriers && marketAccessIntel.access_barriers.length > 0 && (
                          <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                            <h4 className="text-lg font-semibold text-gray-900 mb-3">🚧 Access Barriers</h4>
                            <div className="space-y-2">
                              {marketAccessIntel.access_barriers.slice(0, 5).map((barrier, index) => (
                                <div key={index} className="bg-white p-3 rounded border">
                                  <div className="text-sm font-medium text-gray-900 capitalize">{barrier.type} Barrier</div>
                                  <div className="text-sm text-gray-600">{barrier.description}</div>
                                  <div className={`text-xs mt-1 px-2 py-1 rounded ${
                                    barrier.severity === 'high' ? 'bg-red-100 text-red-800' :
                                    barrier.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-green-100 text-green-800'
                                  }`}>
                                    {barrier.severity} severity
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* HEOR Requirements */}
                        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📊 HEOR Requirements</h4>
                          <div className="text-sm text-gray-700">
                            {marketAccessIntel.heor_requirements?.content || "Health economic evaluation standards and budget impact requirements."}
                          </div>
                        </div>

                        {/* Recommendations */}
                        {marketAccessIntel.recommendations && marketAccessIntel.recommendations.length > 0 && (
                          <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                            <h4 className="text-lg font-semibold text-gray-900 mb-3">💡 Strategic Recommendations</h4>
                            <ul className="space-y-1">
                              {marketAccessIntel.recommendations.slice(0, 5).map((recommendation, index) => (
                                <li key={index} className="text-sm text-gray-700">• {recommendation}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Phase 3: Predictive Analytics Tab */}
                  {activeTab === "predictive" && predictiveAnalytics && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🔮 Predictive Analytics</h3>
                        <p className="text-gray-600">
                          ML-enhanced forecasting for {predictiveAnalytics.therapy_area}
                          {predictiveAnalytics.product_name && <span className="font-medium"> - {predictiveAnalytics.product_name}</span>}
                        </p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-1 bg-indigo-100 text-indigo-800 text-sm rounded">
                            🎯 Model Performance: {(Object.values(predictiveAnalytics.model_performance_metrics || {})[0] || 85).toFixed(0)}%
                          </span>
                          <span className="px-2 py-1 bg-purple-100 text-purple-800 text-sm rounded">
                            📈 Forecast Horizon: 10 years
                          </span>
                        </div>
                      </div>

                      {/* Predictive Key Metrics */}
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
                          <div className="text-sm text-gray-600">Peak Revenue</div>
                          <div className="text-2xl font-bold text-indigo-600">
                            ${predictiveAnalytics.revenue_forecasts?.peak_sales || "1.2B"}
                          </div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                          <div className="text-sm text-gray-600">Market Penetration</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {Object.values(predictiveAnalytics.market_penetration_forecast?.projections || {})[5] || 15}%
                          </div>
                        </div>
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <div className="text-sm text-gray-600">Success Probability</div>
                          <div className="text-2xl font-bold text-green-600">
                            {(predictiveAnalytics.scenario_probabilities?.base * 100 || 70).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                          <div className="text-sm text-gray-600">Risk Level</div>
                          <div className="text-2xl font-bold text-purple-600">
                            {predictiveAnalytics.uncertainty_analysis?.forecast_error ? "Medium" : "Low"}
                          </div>
                        </div>
                      </div>

                      {/* Predictive Analysis Sections */}
                      <div className="space-y-4">
                        {/* Revenue Forecasts */}
                        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">💰 Revenue Forecasts</h4>
                          <div className="grid md:grid-cols-2 gap-4">
                            <div>
                              <div className="text-sm font-medium text-gray-900 mb-2">Annual Projections:</div>
                              {Object.entries(predictiveAnalytics.revenue_forecasts?.annual_projections || {}).slice(0, 5).map(([year, value]) => (
                                <div key={year} className="flex justify-between text-sm">
                                  <span>{year}:</span>
                                  <span className="font-medium">${value}M</span>
                                </div>
                              ))}
                            </div>
                            <div>
                              <div className="text-sm font-medium text-gray-900 mb-2">Peak Sales:</div>
                              <div className="text-lg font-bold text-green-600">
                                ${predictiveAnalytics.revenue_forecasts?.peak_sales || "1,200"}M in {predictiveAnalytics.revenue_forecasts?.peak_year || 2029}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Market Penetration */}
                        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📈 Market Penetration Forecast</h4>
                          <div className="text-sm text-gray-700">
                            {predictiveAnalytics.market_penetration_forecast?.content || "Advanced ML models predict steady market penetration with accelerated adoption post-launch."}
                          </div>
                        </div>

                        {/* Competitive Response */}
                        <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🏆 Competitive Response Modeling</h4>
                          <div className="text-sm text-gray-700">
                            {predictiveAnalytics.competitive_response_modeling?.content || "Predictive models assess likely competitive responses and market dynamics."}
                          </div>
                        </div>

                        {/* Key Assumptions */}
                        {predictiveAnalytics.key_assumptions && predictiveAnalytics.key_assumptions.length > 0 && (
                          <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                            <h4 className="text-lg font-semibold text-gray-900 mb-3">📋 Key Model Assumptions</h4>
                            <ul className="space-y-1">
                              {predictiveAnalytics.key_assumptions.slice(0, 5).map((assumption, index) => (
                                <li key={index} className="text-sm text-gray-700">• {assumption}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Recommendations */}
                        {predictiveAnalytics.recommendations && predictiveAnalytics.recommendations.length > 0 && (
                          <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                            <h4 className="text-lg font-semibold text-gray-900 mb-3">💡 Strategic Recommendations</h4>
                            <ul className="space-y-1">
                              {predictiveAnalytics.recommendations.slice(0, 5).map((recommendation, index) => (
                                <li key={index} className="text-sm text-gray-700">• {recommendation}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Phase 3: Dashboard Tab */}
                  {activeTab === "dashboard" && phase3Dashboard && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">📈 Phase 3 Analytics Dashboard</h3>
                        <p className="text-gray-600">
                          Comprehensive overview of Real-World Evidence, Market Access, and Predictive Analytics
                        </p>
                      </div>

                      {/* Dashboard Summary */}
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <div className="text-sm text-gray-600">RWE Analyses</div>
                          <div className="text-2xl font-bold text-green-600">
                            {phase3Dashboard.summary?.rwe_analyses || 0}
                          </div>
                        </div>
                        <div className="bg-teal-50 p-4 rounded-lg border border-teal-200">
                          <div className="text-sm text-gray-600">Market Access</div>
                          <div className="text-2xl font-bold text-teal-600">
                            {phase3Dashboard.summary?.market_access_analyses || 0}
                          </div>
                        </div>
                        <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
                          <div className="text-sm text-gray-600">Predictive Models</div>
                          <div className="text-2xl font-bold text-indigo-600">
                            {phase3Dashboard.summary?.predictive_analyses || 0}
                          </div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                          <div className="text-sm text-gray-600">Total Analyses</div>
                          <div className="text-2xl font-bold text-purple-600">
                            {phase3Dashboard.summary?.total_phase3_analyses || 0}
                          </div>
                        </div>
                      </div>

                      {/* Capabilities Overview */}
                      <div className="grid md:grid-cols-3 gap-6">
                        <div className="bg-white rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📊 Real-World Evidence</h4>
                          <ul className="space-y-1">
                            {phase3Dashboard.capabilities?.real_world_evidence?.map((capability, index) => (
                              <li key={index} className="text-sm text-gray-600">• {capability}</li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="bg-white rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🏥 Market Access</h4>
                          <ul className="space-y-1">
                            {phase3Dashboard.capabilities?.market_access?.map((capability, index) => (
                              <li key={index} className="text-sm text-gray-600">• {capability}</li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="bg-white rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🔮 Predictive Analytics</h4>
                          <ul className="space-y-1">
                            {phase3Dashboard.capabilities?.predictive_analytics?.map((capability, index) => (
                              <li key={index} className="text-sm text-gray-600">• {capability}</li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Recent Analyses */}
                      <div className="bg-gray-50 rounded-lg p-4 border">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">🕒 Recent Analyses</h4>
                        <div className="grid md:grid-cols-3 gap-4">
                          {/* Recent RWE */}
                          <div>
                            <div className="text-sm font-medium text-gray-900 mb-2">Real-World Evidence</div>
                            {phase3Dashboard.recent_analyses?.rwe?.slice(0, 3).map((analysis, index) => (
                              <div key={index} className="text-xs text-gray-600 mb-1">
                                • {analysis.therapy_area} ({new Date(analysis.timestamp).toLocaleDateString()})
                              </div>
                            ))}
                          </div>
                          
                          {/* Recent Market Access */}
                          <div>
                            <div className="text-sm font-medium text-gray-900 mb-2">Market Access</div>
                            {phase3Dashboard.recent_analyses?.market_access?.slice(0, 3).map((analysis, index) => (
                              <div key={index} className="text-xs text-gray-600 mb-1">
                                • {analysis.therapy_area} ({new Date(analysis.timestamp).toLocaleDateString()})
                              </div>
                            ))}
                          </div>
                          
                          {/* Recent Predictive */}
                          <div>
                            <div className="text-sm font-medium text-gray-900 mb-2">Predictive Analytics</div>
                            {phase3Dashboard.recent_analyses?.predictive?.slice(0, 3).map((analysis, index) => (
                              <div key={index} className="text-xs text-gray-600 mb-1">
                                • {analysis.therapy_area} ({new Date(analysis.timestamp).toLocaleDateString()})
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Phase 4: Executive Dashboard Tab */}
                  {activeTab === "executive" && isAuthenticated && executiveDashboard && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">👑 Executive Dashboard</h3>
                        <p className="text-gray-600">
                          Strategic insights and KPIs for {currentUser?.first_name} at {currentUser?.company || 'your organization'}
                        </p>
                        <div className="flex gap-2 mt-2">
                          <span className="px-2 py-1 bg-purple-100 text-purple-800 text-sm rounded">
                            🎯 Enterprise Level
                          </span>
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded">
                            📊 Real-time Data
                          </span>
                        </div>
                      </div>

                      {/* Executive KPIs */}
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <div className="text-sm text-gray-600">Total Analyses</div>
                          <div className="text-2xl font-bold text-green-600">
                            {executiveDashboard.summary?.total_analyses || 0}
                          </div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                          <div className="text-sm text-gray-600">Success Rate</div>
                          <div className="text-2xl font-bold text-blue-600">
                            {executiveDashboard.trends?.success_rate?.toFixed(1) || 0}%
                          </div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                          <div className="text-sm text-gray-600">Avg Confidence</div>
                          <div className="text-2xl font-bold text-purple-600">
                            {executiveDashboard.trends?.avg_confidence?.toFixed(1) || 0}%
                          </div>
                        </div>
                        <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <div className="text-sm text-gray-600">Time Savings</div>
                          <div className="text-2xl font-bold text-orange-600">
                            {executiveDashboard.trends?.time_savings?.toFixed(1) || 0}%
                          </div>
                        </div>
                      </div>

                      {/* Strategic Analysis Types */}
                      <div className="grid md:grid-cols-3 gap-6">
                        <div className="bg-white rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🔬 Core Analytics</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Therapy Analyses</span>
                              <span className="font-medium">{executiveDashboard.summary?.total_analyses || 0}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Growth Rate</span>
                              <span className="font-medium text-green-600">+{executiveDashboard.trends?.analyses_growth?.toFixed(1) || 0}%</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="bg-white rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">📊 Advanced Features</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">RWE Studies</span>
                              <span className="font-medium">{executiveDashboard.summary?.rwe_analyses || 0}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Market Access</span>
                              <span className="font-medium">{executiveDashboard.summary?.market_access_analyses || 0}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Predictive Models</span>
                              <span className="font-medium">{executiveDashboard.summary?.predictive_analyses || 0}</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="bg-white rounded-lg p-4 border">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">👑 Subscription Status</h4>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Current Plan</span>
                              <span className="font-medium capitalize">{executiveDashboard.subscription?.tier || 'Enterprise'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm text-gray-600">Status</span>
                              <span className={`font-medium ${
                                executiveDashboard.subscription?.status === 'active' ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {executiveDashboard.subscription?.status || 'Active'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Recent Strategic Activity */}
                      <div className="bg-gray-50 rounded-lg p-4 border">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">🕒 Recent Strategic Activity</h4>
                        <div className="space-y-3">
                          {executiveDashboard.recent_activity?.slice(0, 5).map((activity, index) => (
                            <div key={index} className="flex items-center justify-between bg-white p-3 rounded border">
                              <div>
                                <div className="font-medium text-sm">{activity.therapy_area}</div>
                                {activity.product_name && (
                                  <div className="text-xs text-blue-600">{activity.product_name}</div>
                                )}
                              </div>
                              <div className="text-xs text-gray-500">
                                {new Date(activity.created_at).toLocaleDateString()}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Strategic Recommendations */}
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">💡 Strategic Recommendations</h4>
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <h5 className="font-medium text-sm mb-2">Performance Optimization</h5>
                            <ul className="space-y-1 text-sm text-gray-700">
                              <li>• Focus on high-confidence analyses (87%+ confidence)</li>
                              <li>• Expand RWE studies for competitive advantage</li>
                              <li>• Leverage predictive models for strategic planning</li>
                            </ul>
                          </div>
                          <div>
                            <h5 className="font-medium text-sm mb-2">Market Intelligence</h5>
                            <ul className="space-y-1 text-sm text-gray-700">
                              <li>• Monitor competitive landscape changes</li>
                              <li>• Track regulatory pathway developments</li>
                              <li>• Optimize market access strategies</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Welcome Message */}
            {!analysis && !funnel && !competitiveData && !scenarioModels && clinicalTrials.length === 0 && !perplexityResults && !companyIntelligence && !ensembleResult && !financialModel && !timeline && !customTemplate && !advancedViz && !realWorldEvidence && !marketAccessIntel && !predictiveAnalytics && !phase3Dashboard && !executiveDashboard && (
              <div className="bg-white rounded-xl shadow-lg p-8">
                <div className="text-center">
                  <div className="mb-6">
                    <img 
                      src="https://images.unsplash.com/photo-1608222351212-18fe0ec7b13b?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2Njl8MHwxfHNlYXJjaHwxfHxidXNpbmVzcyUyMGFuYWx5dGljc3xlbnwwfHx8fDE3NTI5MjczMjN8MA&ixlib=rb-4.1.0&q=85"
                      alt="Analytics Dashboard"
                      className="mx-auto h-48 w-auto rounded-lg shadow-md mb-6"
                    />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-4">Welcome to Pharma Intelligence Platform</h3>
                  <div className="max-w-2xl mx-auto space-y-4 text-gray-600">
                    <p className="text-lg">
                      Your comprehensive AI-powered pharmaceutical intelligence solution featuring:
                    </p>
                    
                    <div className="grid md:grid-cols-3 gap-4 mt-6">
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg border border-purple-200">
                        <div className="text-2xl mb-2">🤖</div>
                        <h4 className="font-semibold text-gray-900">Multi-Model Ensemble</h4>
                        <p className="text-sm">Claude + Perplexity + Gemini AI with cross-validation & confidence scoring</p>
                      </div>
                      
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">💰</div>
                        <h4 className="font-semibold text-gray-900">Financial Modeling</h4>
                        <p className="text-sm">NPV, IRR, Monte Carlo simulations & sensitivity analysis</p>
                      </div>

                      <div className="bg-green-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">📊</div>
                        <h4 className="font-semibold text-gray-900">Advanced Forecasting</h4>
                        <p className="text-sm">Patient flow funnels, scenario modeling & risk assessment</p>
                      </div>
                      
                      <div className="bg-orange-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">🏢</div>
                        <h4 className="font-semibold text-gray-900">Company Intelligence</h4>
                        <p className="text-sm">Automated competitive research & financial metrics</p>
                      </div>
                      
                      <div className="bg-purple-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">📅</div>
                        <h4 className="font-semibold text-gray-900">Interactive Timelines</h4>
                        <p className="text-sm">Milestone tracking, regulatory pathways & competitive events</p>
                      </div>

                      <div className="bg-teal-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">🔍</div>
                        <h4 className="font-semibold text-gray-900">Clinical Research</h4>
                        <p className="text-sm">Real-time clinical trials data & regulatory intelligence</p>
                      </div>

                      <div className="bg-yellow-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">📋</div>
                        <h4 className="font-semibold text-gray-900">Custom Templates</h4>
                        <p className="text-sm">Therapy-specific, regulatory & KOL interview templates</p>
                      </div>

                      <div className="bg-indigo-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">🎯</div>
                        <h4 className="font-semibold text-gray-900">Advanced Visualizations</h4>
                        <p className="text-sm">2D positioning maps, heatmaps & risk-return analysis</p>
                      </div>

                      <div className="bg-pink-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">📄</div>
                        <h4 className="font-semibold text-gray-900">Export & Reporting</h4>
                        <p className="text-sm">PDF reports, Excel models & presentation-ready outputs</p>
                      </div>

                      {/* Phase 3: New Features */}
                      <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
                        <div className="text-2xl mb-2">🔬</div>
                        <h4 className="font-semibold text-gray-900">Real-World Evidence</h4>
                        <p className="text-sm">RWE analysis, effectiveness studies & patient outcomes</p>
                      </div>

                      <div className="bg-gradient-to-r from-teal-50 to-cyan-50 p-4 rounded-lg border border-teal-200">
                        <div className="text-2xl mb-2">🏥</div>
                        <h4 className="font-semibold text-gray-900">Market Access Intelligence</h4>
                        <p className="text-sm">Payer landscape, reimbursement & regulatory pathways</p>
                      </div>

                      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-4 rounded-lg border border-indigo-200">
                        <div className="text-2xl mb-2">🔮</div>
                        <h4 className="font-semibold text-gray-900">Predictive Analytics</h4>
                        <p className="text-sm">ML-enhanced forecasting & competitive response modeling</p>
                      </div>

                      {/* Phase 4: New Features */}
                      <div className="bg-gradient-to-r from-gold-50 to-yellow-50 p-4 rounded-lg border border-yellow-200">
                        <div className="text-2xl mb-2">🔐</div>
                        <h4 className="font-semibold text-gray-900">User Management & Auth</h4>
                        <p className="text-sm">Secure accounts, session management & user profiles</p>
                      </div>

                      <div className="bg-gradient-to-r from-emerald-50 to-green-50 p-4 rounded-lg border border-emerald-200">
                        <div className="text-2xl mb-2">💳</div>
                        <h4 className="font-semibold text-gray-900">Subscription Plans</h4>
                        <p className="text-sm">Flexible pricing tiers with Stripe payment integration</p>
                      </div>

                      <div className="bg-gradient-to-r from-violet-50 to-purple-50 p-4 rounded-lg border border-violet-200">
                        <div className="text-2xl mb-2">🤖</div>
                        <h4 className="font-semibold text-gray-900">Automated Workflows</h4>
                        <p className="text-sm">Scheduled analysis, smart alerts & automated reporting</p>
                      </div>

                      <div className="bg-gradient-to-r from-rose-50 to-pink-50 p-4 rounded-lg border border-rose-200">
                        <div className="text-2xl mb-2">👑</div>
                        <h4 className="font-semibold text-gray-900">Executive Dashboard</h4>
                        <p className="text-sm">C-level KPIs, strategic insights & performance analytics</p>
                      </div>

                      <div className="bg-gradient-to-r from-sky-50 to-blue-50 p-4 rounded-lg border border-sky-200">
                        <div className="text-2xl mb-2">🔗</div>
                        <h4 className="font-semibold text-gray-900">API & Integrations</h4>
                        <p className="text-sm">REST APIs, webhooks & enterprise system integrations</p>
                      </div>
                    </div>
                    
                    <div className="mt-8 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-900 mb-2">🚀 Ready to Get Started?</p>
                      <p className="text-sm">Enter your Anthropic API key and therapy area to unlock comprehensive pharmaceutical intelligence and forecasting capabilities.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Phase 4: Authentication Modals */}
      {/* Login Modal */}
      {showLoginModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Login to Your Account</h3>
            
            {/* OAuth Buttons */}
            <div className="space-y-3 mb-6">
              <GoogleLogin
                onSuccess={handleGoogleLogin}
                onError={() => setError("Google login failed. Please try again.")}
                theme="outline"
                size="large"
                width="100%"
                text="signin_with"
              />
              
              <div className="w-full">
                <AppleSignin
                  authOptions={{
                    clientId: 'com.pharma.intelligence.platform',
                    scope: 'name email',
                    redirectURI: window.location.origin + '/auth/apple/callback',
                    state: 'pharma_login',
                    nonce: Math.random().toString(36).substring(2, 15),
                    usePopup: true,
                  }}
                  uiType="dark"
                  className="w-full apple-signin-button"
                  onSuccess={handleAppleLogin}
                  onError={() => setError("Apple login failed. Please try again.")}
                />
              </div>
              
              <div className="relative my-4">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="bg-white px-2 text-gray-500">Or continue with email</span>
                </div>
              </div>
            </div>

            <form onSubmit={(e) => {
              e.preventDefault();
              const formData = new FormData(e.target);
              handleLogin({
                email: formData.get('email'),
                password: formData.get('password')
              });
            }}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                  <input
                    name="email"
                    type="email"
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                  <input
                    name="password"
                    type="password"
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowLoginModal(false)}
                  className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loadingStates.login}
                  className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md disabled:bg-gray-300"
                >
                  {loadingStates.login ? 'Logging in...' : 'Login'}
                </button>
              </div>
            </form>
            <p className="text-sm text-center text-gray-600 mt-4">
              Don't have an account?{' '}
              <button
                onClick={() => {
                  setShowLoginModal(false);
                  setShowRegisterModal(true);
                }}
                className="text-blue-600 hover:text-blue-800"
              >
                Sign up here
              </button>
            </p>
          </div>
        </div>
      )}

      {/* Register Modal */}
      {showRegisterModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Create Your Account</h3>
            <form onSubmit={(e) => {
              e.preventDefault();
              const formData = new FormData(e.target);
              handleRegister({
                email: formData.get('email'),
                password: formData.get('password'),
                first_name: formData.get('first_name'),
                last_name: formData.get('last_name'),
                company: formData.get('company'),
                role: formData.get('role')
              });
            }}>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                    <input
                      name="first_name"
                      type="text"
                      required
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                    <input
                      name="last_name"
                      type="text"
                      required
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                  <input
                    name="email"
                    type="email"
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                  <input
                    name="password"
                    type="password"
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Company (Optional)</label>
                  <input
                    name="company"
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Role (Optional)</label>
                  <input
                    name="role"
                    type="text"
                    placeholder="e.g. Market Analyst, Business Development"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowRegisterModal(false)}
                  className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loadingStates.register}
                  className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md disabled:bg-gray-300"
                >
                  {loadingStates.register ? 'Creating Account...' : 'Create Account'}
                </button>
              </div>
            </form>
            <p className="text-sm text-center text-gray-600 mt-4">
              Already have an account?{' '}
              <button
                onClick={() => {
                  setShowRegisterModal(false);
                  setShowLoginModal(true);
                }}
                className="text-blue-600 hover:text-blue-800"
              >
                Login here
              </button>
            </p>
          </div>
        </div>
      )}

      {/* Payment Modal */}
      {showPaymentModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl">
            <h3 className="text-lg font-semibold mb-4">Choose Your Subscription Plan</h3>
            <div className="grid md:grid-cols-3 gap-4">
              {subscriptionPlans.map((plan) => (
                <div key={plan.id} className="border rounded-lg p-4">
                  <h4 className="font-semibold text-lg">{plan.name}</h4>
                  <div className="text-2xl font-bold text-blue-600 my-2">
                    ${plan.price.toFixed(2)}<span className="text-sm text-gray-600">/month</span>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">{plan.description}</p>
                  <ul className="space-y-1 mb-4">
                    {plan.features.map((feature, index) => (
                      <li key={index} className="text-sm text-gray-700 flex items-start">
                        <span className="text-green-500 mr-2">✓</span>
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <button
                    onClick={() => handlePayment(plan.id)}
                    disabled={loadingStates.payment}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md disabled:bg-gray-300"
                  >
                    {loadingStates.payment ? 'Processing...' : 'Subscribe'}
                  </button>
                </div>
              ))}
            </div>
            <div className="flex justify-end mt-6">
              <button
                onClick={() => setShowPaymentModal(false)}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;