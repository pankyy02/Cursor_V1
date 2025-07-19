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
  const [loadingStates, setLoadingStates] = useState({
    analysis: false,
    funnel: false,
    competitive: false,
    scenarios: false,
    trials: false,
    export: false
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

  const setLoadingState = (operation, isLoading) => {
    setLoadingStates(prev => ({
      ...prev,
      [operation]: isLoading
    }));
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
                  {isLoading ? 'Analyzing...' : '🔬 Comprehensive Analysis'}
                </button>

                {analysis && (
                  <>
                    <button
                      onClick={handleGenerateFunnel}
                      disabled={isLoading}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        isLoading 
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-700 text-white'
                      }`}
                    >
                      {isLoading ? 'Generating...' : '📊 Generate Forecast Funnel'}
                    </button>

                    <button
                      onClick={handleCompetitiveAnalysis}
                      disabled={isLoading}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        isLoading 
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-purple-600 hover:bg-purple-700 text-white'
                      }`}
                    >
                      {isLoading ? 'Analyzing...' : '🏆 Competitive Intelligence'}
                    </button>

                    <button
                      onClick={handleScenarioModeling}
                      disabled={isLoading}
                      className={`w-full px-4 py-3 rounded-md font-medium transition-colors ${
                        isLoading 
                          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                      }`}
                    >
                      {isLoading ? 'Modeling...' : '🎯 Scenario Modeling'}
                    </button>

                    <button
                      onClick={handleSearchTrials}
                      disabled={isLoading}
                      className="w-full px-4 py-3 rounded-md font-medium transition-colors bg-teal-600 hover:bg-teal-700 text-white"
                    >
                      🔍 Clinical Trials Research
                    </button>
                  </>
                )}
              </div>

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
            {(analysis || funnel || competitiveData || scenarioModels || clinicalTrials.length > 0) && (
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
                  </nav>
                </div>

                <div className="p-6">
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

                      {/* Visualization */}
                      {funnel.visualization_data && funnel.visualization_data.funnel_chart && (
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">📊 Interactive Funnel Visualization</h4>
                          <div 
                            dangerouslySetInnerHTML={{
                              __html: `<div id="funnel-chart-${funnel.id}"></div>
                              <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                              <script>
                                Plotly.newPlot('funnel-chart-${funnel.id}', ${funnel.visualization_data.funnel_chart});
                              </script>`
                            }}
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
                          <h4 className="text-lg font-semibold text-gray-900 mb-4">📈 Scenario Forecasts</h4>
                          <div 
                            dangerouslySetInnerHTML={{
                              __html: `<div id="scenario-chart-${funnel.id}"></div>
                              <script>
                                Plotly.newPlot('scenario-chart-${funnel.id}', ${funnel.visualization_data.scenario_chart});
                              </script>`
                            }}
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

                  {/* Competitive Intelligence Tab */}
                  {activeTab === "competitive" && competitiveData && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🏆 Competitive Landscape</h3>
                        <p className="text-gray-600">Market intelligence for {therapyArea}</p>
                      </div>

                      {/* Key Competitors */}
                      {competitiveData.competitors && (
                        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
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

                      {/* Market Dynamics */}
                      <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">📊 Market Dynamics</h4>
                        <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">
                          {competitiveData.market_dynamics || competitiveData.full_analysis}
                        </div>
                      </div>

                      {/* Pipeline Analysis */}
                      {competitiveData.pipeline && (
                        <div className="bg-teal-50 rounded-lg p-4 border border-teal-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">🔬 Pipeline Intelligence</h4>
                          <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{competitiveData.pipeline}</div>
                        </div>
                      )}

                      {/* Upcoming Catalysts */}
                      {competitiveData.catalysts && (
                        <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                          <h4 className="text-lg font-semibold text-gray-900 mb-3">⚡ Key Catalysts</h4>
                          <div className="text-gray-700 whitespace-pre-wrap leading-relaxed">{competitiveData.catalysts}</div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Scenario Modeling Tab */}
                  {activeTab === "scenarios" && scenarioModels && (
                    <div className="space-y-6">
                      <div className="mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">🎯 Scenario Modeling & Forecasts</h3>
                        <p className="text-gray-600">Multi-scenario analysis for {therapyArea}</p>
                      </div>

                      {/* Scenario Comparison */}
                      <div className="grid lg:grid-cols-3 gap-4">
                        {Object.entries(scenarioModels).map(([scenario, data]) => (
                          <div key={scenario} className={`rounded-lg p-4 border ${
                            scenario === 'optimistic' ? 'bg-green-50 border-green-200' :
                            scenario === 'pessimistic' ? 'bg-red-50 border-red-200' :
                            'bg-blue-50 border-blue-200'
                          }`}>
                            <h4 className="text-lg font-semibold mb-3 capitalize">
                              {scenario === 'optimistic' ? '📈' : scenario === 'pessimistic' ? '📉' : '📊'} {scenario} Scenario
                            </h4>
                            
                            {data.peak_sales && (
                              <div className="mb-3 p-3 bg-white rounded-md">
                                <div className="text-sm text-gray-600">Peak Sales</div>
                                <div className="text-xl font-bold text-gray-900">${data.peak_sales}M</div>
                              </div>
                            )}

                            {data.projections && (
                              <div className="mb-3">
                                <div className="text-sm text-gray-600 mb-2">6-Year Revenue Projection</div>
                                <div className="space-y-1">
                                  {data.projections.map((projection, index) => (
                                    <div key={index} className="flex justify-between text-sm">
                                      <span>{2024 + index}</span>
                                      <span className="font-medium">${projection}M</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {data.key_factors && (
                              <div className="mt-3">
                                <div className="text-sm text-gray-600 mb-2">Key Success Factors</div>
                                <ul className="text-xs space-y-1">
                                  {data.key_factors.map((factor, index) => (
                                    <li key={index} className="flex items-start">
                                      <span className="mr-1">•</span>
                                      <span>{factor}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        ))}
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
                </div>
              </div>
            )}

            {/* Welcome Message */}
            {!analysis && !funnel && !competitiveData && !scenarioModels && clinicalTrials.length === 0 && (
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
                    
                    <div className="grid md:grid-cols-2 gap-4 mt-6">
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">🔬</div>
                        <h4 className="font-semibold text-gray-900">Deep Therapy Analysis</h4>
                        <p className="text-sm">Disease summaries, biomarkers, treatment algorithms, patient journeys</p>
                      </div>
                      
                      <div className="bg-green-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">📊</div>
                        <h4 className="font-semibold text-gray-900">Advanced Forecasting</h4>
                        <p className="text-sm">Patient flow funnels, scenario modeling, market projections</p>
                      </div>
                      
                      <div className="bg-purple-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">🏆</div>
                        <h4 className="font-semibold text-gray-900">Competitive Intelligence</h4>
                        <p className="text-sm">Market players, pipeline analysis, competitive positioning</p>
                      </div>
                      
                      <div className="bg-teal-50 p-4 rounded-lg">
                        <div className="text-2xl mb-2">🔍</div>
                        <h4 className="font-semibold text-gray-900">Clinical Trials Research</h4>
                        <p className="text-sm">Real-time clinical trial data, regulatory intelligence</p>
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
    </div>
  );
};

export default App;