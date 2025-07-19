import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const App = () => {
  const [apiKey, setApiKey] = useState("");
  const [therapyArea, setTherapyArea] = useState("");
  const [productName, setProductName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [funnel, setFunnel] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("analysis");

  useEffect(() => {
    // Test API connection
    const testConnection = async () => {
      try {
        await axios.get(`${API}/`);
        console.log("API connection successful");
      } catch (e) {
        console.error("API connection failed:", e);
      }
    };
    testConnection();
  }, []);

  const handleAnalyzeTherapy = async () => {
    if (!apiKey.trim()) {
      setError("Please enter your Anthropic API key");
      return;
    }
    if (!therapyArea.trim()) {
      setError("Please enter a therapy area");
      return;
    }

    setIsLoading(true);
    setError("");
    setFunnel(null);

    try {
      const response = await axios.post(`${API}/analyze-therapy`, {
        therapy_area: therapyArea,
        product_name: productName || null,
        api_key: apiKey
      });

      setAnalysis(response.data);
      setActiveTab("analysis");
    } catch (error) {
      console.error("Analysis error:", error);
      setError(error.response?.data?.detail || "Analysis failed. Please check your API key and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateFunnel = async () => {
    if (!analysis || !apiKey.trim()) {
      setError("Please complete therapy area analysis first");
      return;
    }

    setIsLoading(true);
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
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setTherapyArea("");
    setProductName("");
    setAnalysis(null);
    setFunnel(null);
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
              <h1 className="text-xl font-bold text-gray-900">Pharma Forecasting Consultant</h1>
            </div>
            <button
              onClick={resetForm}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
            >
              New Analysis
            </button>
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
            End-to-End Forecasting Consultant
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Comprehensive therapy area analysis, patient journey mapping, and forecasting funnel generation 
            powered by advanced AI for pharmaceutical professionals.
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h3>
              
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

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={handleAnalyzeTherapy}
                  disabled={isLoading}
                  className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                    isLoading 
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Therapy Area'}
                </button>

                {analysis && (
                  <button
                    onClick={handleGenerateFunnel}
                    disabled={isLoading}
                    className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                      isLoading 
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {isLoading ? 'Generating...' : 'Generate Patient Flow Funnel'}
                  </button>
                )}
              </div>

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
            {(analysis || funnel) && (
              <div className="bg-white rounded-xl shadow-lg">
                {/* Tabs */}
                <div className="border-b border-gray-200">
                  <nav className="-mb-px flex">
                    <button
                      onClick={() => setActiveTab("analysis")}
                      className={`px-6 py-3 text-sm font-medium ${
                        activeTab === "analysis"
                          ? 'text-blue-600 border-b-2 border-blue-600'
                          : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      Therapy Analysis
                    </button>
                    {funnel && (
                      <button
                        onClick={() => setActiveTab("funnel")}
                        className={`px-6 py-3 text-sm font-medium ${
                          activeTab === "funnel"
                            ? 'text-blue-600 border-b-2 border-blue-600'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        Patient Flow Funnel
                      </button>
                    )}
                  </nav>
                </div>

                <div className="p-6">
                  {/* Analysis Tab */}
                  {activeTab === "analysis" && analysis && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">
                          {analysis.therapy_area}
                          {analysis.product_name && (
                            <span className="text-lg text-blue-600"> - {analysis.product_name}</span>
                          )}
                        </h3>
                      </div>

                      {[
                        { title: "Disease Summary", content: analysis.disease_summary },
                        { title: "Staging", content: analysis.staging },
                        { title: "Biomarkers", content: analysis.biomarkers },
                        { title: "Treatment Algorithm", content: analysis.treatment_algorithm },
                        { title: "Patient Journey", content: analysis.patient_journey }
                      ].map((section, index) => (
                        <div key={index} className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">{section.title}</h4>
                          <div className="text-gray-700 whitespace-pre-wrap">{section.content}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Funnel Tab */}
                  {activeTab === "funnel" && funnel && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">Patient Flow Funnel</h3>
                        <p className="text-gray-600">Forecasting model for {funnel.therapy_area}</p>
                      </div>

                      {/* Funnel Stages */}
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold text-gray-900 mb-4">Funnel Stages</h4>
                        <div className="space-y-3">
                          {funnel.funnel_stages.map((stage, index) => (
                            <div key={index} className="bg-white rounded-md p-4 border-l-4 border-blue-500">
                              <div className="flex justify-between items-start mb-2">
                                <h5 className="font-medium text-gray-900">{stage.stage}</h5>
                                <span className="text-sm font-bold text-blue-600">{stage.percentage}</span>
                              </div>
                              <p className="text-sm text-gray-600 mb-2">{stage.description}</p>
                              <p className="text-xs text-gray-500">{stage.notes}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* TAM Analysis */}
                      <div className="bg-blue-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Total Addressable Market</h4>
                        <div className="text-gray-700 whitespace-pre-wrap">{funnel.total_addressable_population}</div>
                      </div>

                      {/* Forecasting Notes */}
                      <div className="bg-green-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Forecasting Notes</h4>
                        <div className="text-gray-700 whitespace-pre-wrap">{funnel.forecasting_notes}</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Welcome Message */}
            {!analysis && !funnel && (
              <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <div className="mb-4">
                  <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Analyze</h3>
                <p className="text-gray-600 max-w-md mx-auto">
                  Enter your Anthropic API key and therapy area to begin comprehensive analysis and forecasting funnel generation.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;