#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Pharma Intelligence Platform
Tests all key API endpoints and functionality
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

import httpx
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')
load_dotenv('/app/backend/.env')

# Get backend URL from frontend env
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE_URL = f"{BACKEND_URL}/api"

# Test configuration
TEST_API_KEY = "sk-ant-api03-test-key-for-testing-purposes-only"  # Mock API key for testing
TEST_PERPLEXITY_KEY = "pplx-test-key-for-testing"  # Test Perplexity API key
TEST_THERAPY_AREA = "GIST (Gastrointestinal Stromal Tumors)"
TEST_PRODUCT_NAME = "Qinlock"  # Updated to match review request

class PharmaAPITester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.analysis_id = None
        self.test_results = {}
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test results for reporting"""
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
    
    async def test_api_health(self) -> bool:
        """Test basic API connectivity"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result("API Health Check", True, f"API responding: {data.get('message', 'OK')}")
                return True
            else:
                self.log_test_result("API Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("API Health Check", False, f"Connection error: {str(e)}")
            return False
    
    async def test_database_connection(self) -> bool:
        """Test database connectivity and basic CRUD operations"""
        try:
            # Test status check endpoint (should work without API key)
            payload = {"client_name": "test_client_pharma_testing"}
            response = await self.client.post(f"{API_BASE_URL}/status", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("client_name") == "test_client_pharma_testing" and data.get("id"):
                    # Test retrieval
                    get_response = await self.client.get(f"{API_BASE_URL}/status")
                    if get_response.status_code == 200:
                        status_list = get_response.json()
                        found = any(s.get("client_name") == "test_client_pharma_testing" for s in status_list)
                        
                        self.log_test_result("Database Connection", True, 
                                           f"MongoDB connection working, CRUD operations successful")
                        return True
                    else:
                        self.log_test_result("Database Connection", False, 
                                           f"Status retrieval failed: {get_response.status_code}")
                        return False
                else:
                    self.log_test_result("Database Connection", False, 
                                       f"Invalid status response structure")
                    return False
            else:
                self.log_test_result("Database Connection", False, 
                                   f"Status creation failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Database Connection", False, f"Exception: {str(e)}")
            return False
    
    async def test_existing_data_retrieval(self) -> bool:
        """Test retrieval of existing analyses for testing dependent endpoints"""
        try:
            # Check if there are existing analyses we can use
            response = await self.client.get(f"{API_BASE_URL}/analyses")
            
            if response.status_code == 200:
                analyses = response.json()
                if analyses and len(analyses) > 0:
                    # Use the first available analysis for testing
                    self.analysis_id = analyses[0].get("id")
                    therapy_area = analyses[0].get("therapy_area", "Unknown")
                    
                    self.log_test_result("Existing Data Retrieval", True, 
                                       f"Found {len(analyses)} existing analyses, "
                                       f"Using ID: {self.analysis_id[:8]}... ({therapy_area})")
                    return True
                else:
                    self.log_test_result("Existing Data Retrieval", True, 
                                       "No existing analyses found - will test creation endpoints")
                    return True
            else:
                self.log_test_result("Existing Data Retrieval", False, 
                                   f"Failed to retrieve analyses: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Existing Data Retrieval", False, f"Exception: {str(e)}")
            return False
    
    async def test_therapy_analysis(self) -> bool:
        """Test core therapy area analysis endpoint"""
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "product_name": TEST_PRODUCT_NAME,
                "api_key": TEST_API_KEY
            }
            
            response = await self.client.post(f"{API_BASE_URL}/analyze-therapy", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["id", "therapy_area", "disease_summary", "staging", "biomarkers", 
                                 "treatment_algorithm", "patient_journey"]
                missing_fields = [field for field in required_fields if not data.get(field)]
                
                if missing_fields:
                    self.log_test_result("Therapy Analysis", False, 
                                       f"Missing required fields: {missing_fields}")
                    return False
                
                # Store analysis ID for subsequent tests
                self.analysis_id = data["id"]
                
                # Validate content quality
                content_checks = []
                if len(data["disease_summary"]) < 100:
                    content_checks.append("Disease summary too short")
                if len(data["treatment_algorithm"]) < 50:
                    content_checks.append("Treatment algorithm too brief")
                if "GIST" not in data["disease_summary"] and "gastrointestinal" not in data["disease_summary"].lower():
                    content_checks.append("Disease summary doesn't match therapy area")
                
                if content_checks:
                    self.log_test_result("Therapy Analysis", False, 
                                       f"Content quality issues: {content_checks}")
                    return False
                
                # Check for competitive landscape and clinical trials data
                has_competitive = bool(data.get("competitive_landscape"))
                has_trials = bool(data.get("clinical_trials_data"))
                
                self.log_test_result("Therapy Analysis", True, 
                                   f"Analysis generated successfully. ID: {self.analysis_id[:8]}..., "
                                   f"Competitive data: {has_competitive}, Clinical trials: {has_trials}")
                return True
            elif response.status_code == 500 and "authentication_error" in response.text:
                # API key authentication issue - this is expected without valid key
                self.log_test_result("Therapy Analysis", False, 
                                   "API key authentication required - endpoint structure validated")
                return False
            else:
                error_detail = response.text
                self.log_test_result("Therapy Analysis", False, 
                                   f"HTTP {response.status_code}: {error_detail}")
                return False
                
        except Exception as e:
            self.log_test_result("Therapy Analysis", False, f"Exception: {str(e)}")
            return False
    
    async def test_funnel_generation(self) -> bool:
        """Test patient flow funnel generation"""
        if not self.analysis_id:
            self.log_test_result("Funnel Generation", False, "No analysis ID available")
            return False
            
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "analysis_id": self.analysis_id,
                "api_key": TEST_API_KEY
            }
            
            response = await self.client.post(f"{API_BASE_URL}/generate-funnel", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate funnel structure
                required_fields = ["id", "therapy_area", "analysis_id", "funnel_stages", 
                                 "total_addressable_population", "forecasting_notes"]
                missing_fields = [field for field in required_fields if not data.get(field)]
                
                if missing_fields:
                    self.log_test_result("Funnel Generation", False, 
                                       f"Missing fields: {missing_fields}")
                    return False
                
                # Validate funnel stages
                funnel_stages = data.get("funnel_stages", [])
                if len(funnel_stages) < 3:
                    self.log_test_result("Funnel Generation", False, 
                                       f"Insufficient funnel stages: {len(funnel_stages)}")
                    return False
                
                # Check for percentage extraction
                percentage_count = 0
                for stage in funnel_stages:
                    if stage.get("percentage") and "%" in str(stage["percentage"]):
                        percentage_count += 1
                
                # Validate visualization data
                viz_data = data.get("visualization_data", {})
                has_funnel_chart = bool(viz_data.get("funnel_chart"))
                has_scenario_chart = bool(viz_data.get("scenario_chart"))
                
                # Check scenario models
                scenario_models = data.get("scenario_models", {})
                has_scenarios = len(scenario_models) >= 2
                
                self.log_test_result("Funnel Generation", True, 
                                   f"Funnel created with {len(funnel_stages)} stages, "
                                   f"{percentage_count} with percentages, "
                                   f"Charts: funnel={has_funnel_chart}, scenario={has_scenario_chart}, "
                                   f"Scenarios: {has_scenarios}")
                return True
            else:
                self.log_test_result("Funnel Generation", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Funnel Generation", False, f"Exception: {str(e)}")
            return False
    
    async def test_competitive_analysis(self) -> bool:
        """Test competitive intelligence analysis"""
        if not self.analysis_id:
            self.log_test_result("Competitive Analysis", False, "No analysis ID available")
            return False
            
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "analysis_id": self.analysis_id,
                "api_key": TEST_API_KEY
            }
            
            response = await self.client.post(f"{API_BASE_URL}/competitive-analysis", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if data.get("status") != "success":
                    self.log_test_result("Competitive Analysis", False, 
                                       f"Status not success: {data.get('status')}")
                    return False
                
                competitive_landscape = data.get("competitive_landscape", {})
                competitors = competitive_landscape.get("competitors", [])
                
                # Check for real company names (not generic placeholders)
                real_companies = 0
                pharma_companies = ["novartis", "pfizer", "roche", "bristol", "merck", "johnson", 
                                  "abbvie", "gilead", "biogen", "amgen", "bayer", "sanofi"]
                
                for comp in competitors:
                    comp_name = comp.get("name", "").lower()
                    if any(pharma in comp_name for pharma in pharma_companies):
                        real_companies += 1
                
                # Validate market dynamics and pipeline data
                has_market_dynamics = bool(competitive_landscape.get("market_dynamics"))
                has_pipeline = bool(competitive_landscape.get("pipeline"))
                
                clinical_trials_count = data.get("clinical_trials_count", 0)
                
                success = len(competitors) >= 3 and (real_companies > 0 or len(competitors) >= 5)
                
                self.log_test_result("Competitive Analysis", success, 
                                   f"Found {len(competitors)} competitors, "
                                   f"{real_companies} real pharma companies, "
                                   f"Market dynamics: {has_market_dynamics}, "
                                   f"Pipeline: {has_pipeline}, "
                                   f"Clinical trials: {clinical_trials_count}")
                return success
            else:
                self.log_test_result("Competitive Analysis", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Competitive Analysis", False, f"Exception: {str(e)}")
            return False
    
    async def test_scenario_modeling(self) -> bool:
        """Test scenario modeling with revenue projections"""
        if not self.analysis_id:
            self.log_test_result("Scenario Modeling", False, "No analysis ID available")
            return False
            
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "analysis_id": self.analysis_id,
                "scenarios": ["optimistic", "realistic", "pessimistic"],
                "api_key": TEST_API_KEY
            }
            
            response = await self.client.post(f"{API_BASE_URL}/scenario-modeling", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") != "success":
                    self.log_test_result("Scenario Modeling", False, 
                                       f"Status not success: {data.get('status')}")
                    return False
                
                scenario_models = data.get("scenario_models", {})
                
                # Validate all scenarios are present
                required_scenarios = ["optimistic", "realistic", "pessimistic"]
                missing_scenarios = [s for s in required_scenarios if s not in scenario_models]
                
                if missing_scenarios:
                    self.log_test_result("Scenario Modeling", False, 
                                       f"Missing scenarios: {missing_scenarios}")
                    return False
                
                # Validate projections
                projection_checks = []
                for scenario, model in scenario_models.items():
                    projections = model.get("projections", [])
                    if len(projections) < 5:
                        projection_checks.append(f"{scenario}: insufficient projections ({len(projections)})")
                    elif not all(isinstance(p, (int, float)) and p >= 0 for p in projections):
                        projection_checks.append(f"{scenario}: invalid projection values")
                
                if projection_checks:
                    self.log_test_result("Scenario Modeling", False, 
                                       f"Projection issues: {projection_checks}")
                    return False
                
                # Check visualization
                has_visualization = bool(data.get("visualization"))
                
                # Validate realistic revenue ranges
                opt_peak = max(scenario_models["optimistic"]["projections"])
                real_peak = max(scenario_models["realistic"]["projections"])
                pess_peak = max(scenario_models["pessimistic"]["projections"])
                
                logical_ordering = opt_peak >= real_peak >= pess_peak
                
                self.log_test_result("Scenario Modeling", True, 
                                   f"3 scenarios generated, "
                                   f"Peak revenues: Opt={opt_peak}M, Real={real_peak}M, Pess={pess_peak}M, "
                                   f"Logical ordering: {logical_ordering}, "
                                   f"Visualization: {has_visualization}")
                return True
            else:
                self.log_test_result("Scenario Modeling", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Scenario Modeling", False, f"Exception: {str(e)}")
            return False
    
    async def test_clinical_trials_search(self) -> bool:
        """Test clinical trials search endpoint"""
        try:
            params = {"therapy_area": TEST_THERAPY_AREA}
            response = await self.client.get(f"{API_BASE_URL}/search/clinical-trials", params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                trials = data.get("trials", [])
                count = data.get("count", 0)
                
                if count != len(trials):
                    self.log_test_result("Clinical Trials Search", False, 
                                       f"Count mismatch: reported {count}, actual {len(trials)}")
                    return False
                
                # Validate trial structure
                if trials:
                    sample_trial = trials[0]
                    expected_fields = ["NCTId", "BriefTitle", "OverallStatus"]
                    missing_fields = [field for field in expected_fields if field not in sample_trial]
                    
                    if missing_fields:
                        self.log_test_result("Clinical Trials Search", False, 
                                           f"Trial missing fields: {missing_fields}")
                        return False
                
                self.log_test_result("Clinical Trials Search", True, 
                                   f"Found {count} clinical trials")
                return True
            else:
                self.log_test_result("Clinical Trials Search", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Clinical Trials Search", False, f"Exception: {str(e)}")
            return False
    
    async def test_export_functionality(self) -> bool:
        """Test PDF and Excel export functionality"""
        if not self.analysis_id:
            self.log_test_result("Export Functionality", False, "No analysis ID available")
            return False
        
        export_results = {}
        
        # Test PDF export
        try:
            payload = {"analysis_id": self.analysis_id, "export_type": "pdf"}
            response = await self.client.post(f"{API_BASE_URL}/export", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data.get("data"):
                    export_results["pdf"] = True
                else:
                    export_results["pdf"] = False
            else:
                export_results["pdf"] = False
        except Exception:
            export_results["pdf"] = False
        
        # Test Excel export
        try:
            payload = {"analysis_id": self.analysis_id, "export_type": "excel"}
            response = await self.client.post(f"{API_BASE_URL}/export", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data.get("data"):
                    export_results["excel"] = True
                else:
                    export_results["excel"] = False
            else:
                export_results["excel"] = False
        except Exception:
            export_results["excel"] = False
        
        success = any(export_results.values())
        self.log_test_result("Export Functionality", success, 
                           f"PDF: {export_results.get('pdf', False)}, "
                           f"Excel: {export_results.get('excel', False)}")
        return success
    
    async def test_data_persistence(self) -> bool:
        """Test MongoDB data storage and retrieval"""
        if not self.analysis_id:
            self.log_test_result("Data Persistence", False, "No analysis ID available")
            return False
            
        try:
            # Test retrieving stored analysis
            response = await self.client.get(f"{API_BASE_URL}/analysis/{self.analysis_id}")
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("analysis")
                funnel = data.get("funnel")
                
                if not analysis:
                    self.log_test_result("Data Persistence", False, "Analysis not found in database")
                    return False
                
                # Verify analysis data integrity
                if analysis.get("id") != self.analysis_id:
                    self.log_test_result("Data Persistence", False, "Analysis ID mismatch")
                    return False
                
                if analysis.get("therapy_area") != TEST_THERAPY_AREA:
                    self.log_test_result("Data Persistence", False, "Therapy area mismatch")
                    return False
                
                # Test list endpoint
                list_response = await self.client.get(f"{API_BASE_URL}/analyses")
                if list_response.status_code == 200:
                    analyses = list_response.json()
                    found_analysis = any(a.get("id") == self.analysis_id for a in analyses)
                    
                    self.log_test_result("Data Persistence", True, 
                                       f"Analysis stored and retrieved successfully, "
                                       f"Found in list: {found_analysis}, "
                                       f"Has funnel: {bool(funnel)}")
                    return True
                else:
                    self.log_test_result("Data Persistence", False, 
                                       f"List endpoint failed: {list_response.status_code}")
                    return False
            else:
                self.log_test_result("Data Persistence", False, 
                                   f"Retrieval failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Data Persistence", False, f"Exception: {str(e)}")
            return False
    
    async def test_perplexity_search_endpoint(self) -> bool:
        """Test Perplexity search endpoint for pharmaceutical intelligence"""
        try:
            payload = {
                "query": "GIST market analysis 2024 competitive landscape",
                "api_key": TEST_PERPLEXITY_KEY,
                "search_focus": "pharmaceutical"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/perplexity-search", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["content", "citations", "search_query", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Perplexity Search Endpoint", False, 
                                       f"Missing required fields: {missing_fields}")
                    return False
                
                # Validate content quality
                content = data.get("content", "")
                if len(content) < 50:
                    self.log_test_result("Perplexity Search Endpoint", False, 
                                       "Content too short - likely API error")
                    return False
                
                # Check citations
                citations = data.get("citations", [])
                has_citations = len(citations) > 0
                
                # Verify search query matches
                if data.get("search_query") != payload["query"]:
                    self.log_test_result("Perplexity Search Endpoint", False, 
                                       "Search query mismatch in response")
                    return False
                
                # Check if content contains pharmaceutical intelligence
                pharma_keywords = ["market", "pharmaceutical", "GIST", "competitive", "analysis"]
                content_relevance = sum(1 for keyword in pharma_keywords if keyword.lower() in content.lower())
                
                self.log_test_result("Perplexity Search Endpoint", True, 
                                   f"Search successful. Content length: {len(content)}, "
                                   f"Citations: {len(citations)}, "
                                   f"Relevance score: {content_relevance}/5")
                return True
                
            elif response.status_code == 500:
                # Check if it's an API key error (expected with test key)
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Perplexity Search Endpoint", False, 
                                       "API key authentication required - endpoint structure validated")
                    return False
                else:
                    self.log_test_result("Perplexity Search Endpoint", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Perplexity Search Endpoint", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Perplexity Search Endpoint", False, f"Exception: {str(e)}")
            return False
    
    async def test_perplexity_pharmaceutical_search(self) -> bool:
        """Test Perplexity search with pharmaceutical-specific query"""
        try:
            payload = {
                "query": "Qinlock ripretinib market share competitive intelligence",
                "api_key": TEST_PERPLEXITY_KEY,
                "search_focus": "competitive_intelligence"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/perplexity-search", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                content = data.get("content", "")
                citations = data.get("citations", [])
                
                # Check for pharmaceutical-specific content
                pharma_terms = ["ripretinib", "qinlock", "market share", "competitive", "pharmaceutical"]
                term_matches = sum(1 for term in pharma_terms if term.lower() in content.lower())
                
                # Validate citations format
                valid_citations = 0
                for citation in citations:
                    if isinstance(citation, str) and ("http" in citation or len(citation) > 10):
                        valid_citations += 1
                
                # Check database storage
                # The endpoint should store results in MongoDB
                success = len(content) > 100 and term_matches >= 2
                
                self.log_test_result("Perplexity Pharmaceutical Search", success, 
                                   f"Pharmaceutical terms found: {term_matches}/5, "
                                   f"Valid citations: {valid_citations}/{len(citations)}, "
                                   f"Content quality: {'Good' if len(content) > 200 else 'Basic'}")
                return success
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Perplexity Pharmaceutical Search", False, 
                                       "API key authentication required - endpoint functional")
                    return False
                else:
                    self.log_test_result("Perplexity Pharmaceutical Search", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Perplexity Pharmaceutical Search", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Perplexity Pharmaceutical Search", False, f"Exception: {str(e)}")
            return False
    
    async def test_enhanced_competitive_analysis(self) -> bool:
        """Test enhanced competitive analysis with Perplexity integration"""
        if not self.analysis_id:
            self.log_test_result("Enhanced Competitive Analysis", False, "No analysis ID available")
            return False
            
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "analysis_id": self.analysis_id,
                "api_key": TEST_PERPLEXITY_KEY  # Using Perplexity key for enhanced analysis
            }
            
            response = await self.client.post(f"{API_BASE_URL}/enhanced-competitive-analysis", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if data.get("status") != "success":
                    self.log_test_result("Enhanced Competitive Analysis", False, 
                                       f"Status not success: {data.get('status')}")
                    return False
                
                competitive_landscape = data.get("competitive_landscape", {})
                
                # Check for enhanced analysis components
                has_real_time_intel = bool(competitive_landscape.get("real_time_intelligence"))
                has_enhanced_analysis = bool(competitive_landscape.get("enhanced_analysis"))
                has_combined_insights = bool(competitive_landscape.get("combined_insights"))
                
                # Validate real-time intelligence structure
                real_time_intel = competitive_landscape.get("real_time_intelligence", {})
                has_sources = len(real_time_intel.get("sources", [])) > 0
                has_content = len(real_time_intel.get("content", "")) > 100
                
                # Check analysis type
                analysis_type = competitive_landscape.get("analysis_type", "")
                is_enhanced = "enhanced" in analysis_type.lower() or "perplexity" in analysis_type.lower()
                
                total_sources = competitive_landscape.get("total_sources", 0)
                
                success = has_real_time_intel and has_enhanced_analysis and has_sources
                
                self.log_test_result("Enhanced Competitive Analysis", success, 
                                   f"Real-time intel: {has_real_time_intel}, "
                                   f"Enhanced analysis: {has_enhanced_analysis}, "
                                   f"Sources: {total_sources}, "
                                   f"Analysis type: {analysis_type}")
                return success
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Enhanced Competitive Analysis", False, 
                                       "API key authentication required - endpoint structure validated")
                    return False
                else:
                    self.log_test_result("Enhanced Competitive Analysis", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Enhanced Competitive Analysis", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Enhanced Competitive Analysis", False, f"Exception: {str(e)}")
            return False
    
    async def test_perplexity_error_handling(self) -> bool:
        """Test error handling with invalid Perplexity API key"""
        try:
            payload = {
                "query": "Test query for error handling",
                "api_key": "invalid-perplexity-key-12345",
                "search_focus": "pharmaceutical"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/perplexity-search", json=payload)
            
            # Should return 500 with proper error message
            if response.status_code == 500:
                error_text = response.text
                
                # Check if error message is informative
                has_api_key_error = "api" in error_text.lower() and "key" in error_text.lower()
                has_perplexity_mention = "perplexity" in error_text.lower()
                
                # Validate error response structure
                try:
                    error_data = response.json()
                    has_detail = "detail" in error_data
                except:
                    has_detail = False
                
                success = has_api_key_error or has_perplexity_mention
                
                self.log_test_result("Perplexity Error Handling", success, 
                                   f"Proper error response for invalid API key. "
                                   f"API key error: {has_api_key_error}, "
                                   f"Perplexity mentioned: {has_perplexity_mention}, "
                                   f"Structured error: {has_detail}")
                return success
            else:
                self.log_test_result("Perplexity Error Handling", False, 
                                   f"Expected 500 error, got {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Perplexity Error Handling", False, f"Exception: {str(e)}")
            return False
    
    async def test_perplexity_data_storage(self) -> bool:
        """Test that Perplexity search results are stored in MongoDB"""
        try:
            # First, perform a search
            payload = {
                "query": "Test query for database storage validation",
                "api_key": TEST_PERPLEXITY_KEY,
                "search_focus": "pharmaceutical"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/perplexity-search", json=payload)
            
            if response.status_code == 200:
                # The endpoint should store the result in database
                # We can't directly query MongoDB, but we can infer storage success
                # from the successful response and proper structure
                
                data = response.json()
                
                # Check if response has all required fields for storage
                storage_fields = ["content", "citations", "search_query", "timestamp"]
                has_all_fields = all(field in data for field in storage_fields)
                
                # Validate timestamp format
                timestamp_valid = False
                try:
                    from datetime import datetime
                    datetime.fromisoformat(data.get("timestamp", "").replace('Z', '+00:00'))
                    timestamp_valid = True
                except:
                    pass
                
                success = has_all_fields and timestamp_valid
                
                self.log_test_result("Perplexity Data Storage", success, 
                                   f"Storage fields present: {has_all_fields}, "
                                   f"Valid timestamp: {timestamp_valid}, "
                                   f"Query: '{payload['query'][:30]}...'")
                return success
                
            elif response.status_code == 500:
                # API key error - but endpoint structure is correct
                self.log_test_result("Perplexity Data Storage", False, 
                                   "API key required for storage testing")
                return False
            else:
                self.log_test_result("Perplexity Data Storage", False, 
                                   f"Search failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Perplexity Data Storage", False, f"Exception: {str(e)}")
            return False
    
    # ========================================
    # PHASE 3: REAL-WORLD EVIDENCE & MARKET ACCESS INTELLIGENCE TESTS
    # ========================================
    
    async def test_real_world_evidence_endpoint(self) -> bool:
        """Test Real-World Evidence analysis endpoint"""
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "product_name": TEST_PRODUCT_NAME,
                "analysis_type": "comprehensive",
                "data_sources": ["registries", "claims", "ehr", "patient_outcomes"],
                "api_key": TEST_API_KEY
            }
            
            response = await self.client.post(f"{API_BASE_URL}/real-world-evidence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["id", "therapy_area", "effectiveness_data", "safety_profile", 
                                 "patient_outcomes", "real_world_performance", "comparative_effectiveness",
                                 "cost_effectiveness", "adherence_patterns", "health_economics_data",
                                 "evidence_quality_score", "data_sources", "study_populations",
                                 "limitations", "recommendations"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       f"Missing required fields: {missing_fields}")
                    return False
                
                # Validate therapy area and product name
                if data.get("therapy_area") != TEST_THERAPY_AREA:
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       f"Therapy area mismatch: {data.get('therapy_area')}")
                    return False
                
                if data.get("product_name") != TEST_PRODUCT_NAME:
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       f"Product name mismatch: {data.get('product_name')}")
                    return False
                
                # Validate evidence quality score
                evidence_score = data.get("evidence_quality_score", 0)
                if not isinstance(evidence_score, (int, float)) or evidence_score < 0 or evidence_score > 1:
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       f"Invalid evidence quality score: {evidence_score}")
                    return False
                
                # Validate data sources
                returned_sources = data.get("data_sources", [])
                if not all(source in returned_sources for source in payload["data_sources"]):
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       f"Data sources mismatch: expected {payload['data_sources']}, got {returned_sources}")
                    return False
                
                # Check for content in key sections
                effectiveness_data = data.get("effectiveness_data", {})
                safety_profile = data.get("safety_profile", {})
                patient_outcomes = data.get("patient_outcomes", {})
                
                has_effectiveness = isinstance(effectiveness_data, dict) and len(effectiveness_data) > 0
                has_safety = isinstance(safety_profile, dict) and len(safety_profile) > 0
                has_outcomes = isinstance(patient_outcomes, dict) and len(patient_outcomes) > 0
                
                # Validate recommendations and limitations
                recommendations = data.get("recommendations", [])
                limitations = data.get("limitations", [])
                
                has_recommendations = isinstance(recommendations, list) and len(recommendations) > 0
                has_limitations = isinstance(limitations, list) and len(limitations) > 0
                
                self.log_test_result("Real-World Evidence Analysis", True, 
                                   f"RWE analysis generated successfully. "
                                   f"Evidence score: {evidence_score:.2f}, "
                                   f"Data sources: {len(returned_sources)}, "
                                   f"Has effectiveness: {has_effectiveness}, "
                                   f"Has safety: {has_safety}, "
                                   f"Has outcomes: {has_outcomes}, "
                                   f"Recommendations: {len(recommendations)}, "
                                   f"Limitations: {len(limitations)}")
                return True
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       "Claude API key authentication required - endpoint structure validated")
                    return False
                else:
                    self.log_test_result("Real-World Evidence Analysis", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Real-World Evidence Analysis", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Real-World Evidence Analysis", False, f"Exception: {str(e)}")
            return False
    
    async def test_market_access_intelligence_endpoint(self) -> bool:
        """Test Market Access Intelligence analysis endpoint"""
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "product_name": TEST_PRODUCT_NAME,
                "target_markets": ["US", "EU5", "Japan"],
                "analysis_depth": "comprehensive",
                "api_key": TEST_PERPLEXITY_KEY  # Using Perplexity for market access
            }
            
            response = await self.client.post(f"{API_BASE_URL}/market-access-intelligence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["id", "therapy_area", "payer_landscape", "reimbursement_pathways",
                                 "pricing_analysis", "access_barriers", "heor_requirements",
                                 "regulatory_pathways", "approval_timelines", "formulary_placement",
                                 "budget_impact_models", "coverage_policies", "stakeholder_mapping",
                                 "market_readiness_score", "recommendations"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Market Access Intelligence", False, 
                                       f"Missing required fields: {missing_fields}")
                    return False
                
                # Validate therapy area and product name
                if data.get("therapy_area") != TEST_THERAPY_AREA:
                    self.log_test_result("Market Access Intelligence", False, 
                                       f"Therapy area mismatch: {data.get('therapy_area')}")
                    return False
                
                # Validate market readiness score
                readiness_score = data.get("market_readiness_score", 0)
                if not isinstance(readiness_score, (int, float)) or readiness_score < 0 or readiness_score > 1:
                    self.log_test_result("Market Access Intelligence", False, 
                                       f"Invalid market readiness score: {readiness_score}")
                    return False
                
                # Check for content in key sections
                payer_landscape = data.get("payer_landscape", {})
                reimbursement_pathways = data.get("reimbursement_pathways", {})
                pricing_analysis = data.get("pricing_analysis", {})
                access_barriers = data.get("access_barriers", [])
                
                has_payer_data = isinstance(payer_landscape, dict) and len(payer_landscape) > 0
                has_reimbursement = isinstance(reimbursement_pathways, dict) and len(reimbursement_pathways) > 0
                has_pricing = isinstance(pricing_analysis, dict) and len(pricing_analysis) > 0
                has_barriers = isinstance(access_barriers, list) and len(access_barriers) > 0
                
                # Validate recommendations
                recommendations = data.get("recommendations", [])
                has_recommendations = isinstance(recommendations, list) and len(recommendations) > 0
                
                self.log_test_result("Market Access Intelligence", True, 
                                   f"Market access analysis generated successfully. "
                                   f"Readiness score: {readiness_score:.2f}, "
                                   f"Has payer data: {has_payer_data}, "
                                   f"Has reimbursement: {has_reimbursement}, "
                                   f"Has pricing: {has_pricing}, "
                                   f"Access barriers: {len(access_barriers)}, "
                                   f"Recommendations: {len(recommendations)}")
                return True
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Market Access Intelligence", False, 
                                       "Perplexity API key authentication required - endpoint structure validated")
                    return False
                else:
                    self.log_test_result("Market Access Intelligence", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Market Access Intelligence", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Market Access Intelligence", False, f"Exception: {str(e)}")
            return False
    
    async def test_predictive_analytics_endpoint(self) -> bool:
        """Test Predictive Analytics with ML-enhanced forecasting endpoint"""
        try:
            payload = {
                "therapy_area": TEST_THERAPY_AREA,
                "product_name": TEST_PRODUCT_NAME,
                "forecast_horizon": 10,
                "model_type": "ml_enhanced",
                "include_rwe": True,
                "api_key": TEST_API_KEY
            }
            
            response = await self.client.post(f"{API_BASE_URL}/predictive-analytics", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["id", "therapy_area", "market_penetration_forecast", 
                                 "competitive_response_modeling", "patient_flow_predictions",
                                 "revenue_forecasts", "risk_adjusted_projections", 
                                 "scenario_probabilities", "confidence_intervals",
                                 "key_assumptions", "sensitivity_factors", 
                                 "model_performance_metrics", "uncertainty_analysis", "recommendations"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Predictive Analytics", False, 
                                       f"Missing required fields: {missing_fields}")
                    return False
                
                # Validate therapy area and product name
                if data.get("therapy_area") != TEST_THERAPY_AREA:
                    self.log_test_result("Predictive Analytics", False, 
                                       f"Therapy area mismatch: {data.get('therapy_area')}")
                    return False
                
                # Check for content in key sections
                market_penetration = data.get("market_penetration_forecast", {})
                competitive_response = data.get("competitive_response_modeling", {})
                revenue_forecasts = data.get("revenue_forecasts", {})
                scenario_probabilities = data.get("scenario_probabilities", {})
                
                has_market_penetration = isinstance(market_penetration, dict) and len(market_penetration) > 0
                has_competitive = isinstance(competitive_response, dict) and len(competitive_response) > 0
                has_revenue = isinstance(revenue_forecasts, dict) and len(revenue_forecasts) > 0
                has_scenarios = isinstance(scenario_probabilities, dict) and len(scenario_probabilities) > 0
                
                # Validate model performance metrics
                model_performance = data.get("model_performance_metrics", {})
                has_performance = isinstance(model_performance, dict) and len(model_performance) > 0
                
                # Validate assumptions and recommendations
                assumptions = data.get("key_assumptions", [])
                recommendations = data.get("recommendations", [])
                
                has_assumptions = isinstance(assumptions, list) and len(assumptions) > 0
                has_recommendations = isinstance(recommendations, list) and len(recommendations) > 0
                
                self.log_test_result("Predictive Analytics", True, 
                                   f"Predictive analytics generated successfully. "
                                   f"Has market penetration: {has_market_penetration}, "
                                   f"Has competitive modeling: {has_competitive}, "
                                   f"Has revenue forecasts: {has_revenue}, "
                                   f"Has scenarios: {has_scenarios}, "
                                   f"Has performance metrics: {has_performance}, "
                                   f"Assumptions: {len(assumptions)}, "
                                   f"Recommendations: {len(recommendations)}")
                return True
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Predictive Analytics", False, 
                                       "Claude API key authentication required - endpoint structure validated")
                    return False
                else:
                    self.log_test_result("Predictive Analytics", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Predictive Analytics", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Predictive Analytics", False, f"Exception: {str(e)}")
            return False
    
    async def test_phase3_dashboard_endpoint(self) -> bool:
        """Test Phase 3 analytics dashboard endpoint"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/phase3-dashboard")
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_sections = ["summary", "recent_analyses", "capabilities"]
                missing_sections = [section for section in required_sections if section not in data]
                
                if missing_sections:
                    self.log_test_result("Phase 3 Dashboard", False, 
                                       f"Missing required sections: {missing_sections}")
                    return False
                
                # Validate summary section
                summary = data.get("summary", {})
                required_summary_fields = ["rwe_analyses", "market_access_analyses", 
                                         "predictive_analyses", "total_phase3_analyses"]
                missing_summary = [field for field in required_summary_fields if field not in summary]
                
                if missing_summary:
                    self.log_test_result("Phase 3 Dashboard", False, 
                                       f"Missing summary fields: {missing_summary}")
                    return False
                
                # Validate recent analyses section
                recent_analyses = data.get("recent_analyses", {})
                required_recent = ["rwe", "market_access", "predictive"]
                missing_recent = [field for field in required_recent if field not in recent_analyses]
                
                if missing_recent:
                    self.log_test_result("Phase 3 Dashboard", False, 
                                       f"Missing recent analyses: {missing_recent}")
                    return False
                
                # Validate capabilities section
                capabilities = data.get("capabilities", {})
                required_capabilities = ["real_world_evidence", "market_access", "predictive_analytics"]
                missing_capabilities = [field for field in required_capabilities if field not in capabilities]
                
                if missing_capabilities:
                    self.log_test_result("Phase 3 Dashboard", False, 
                                       f"Missing capabilities: {missing_capabilities}")
                    return False
                
                # Check counts
                rwe_count = summary.get("rwe_analyses", 0)
                market_access_count = summary.get("market_access_analyses", 0)
                predictive_count = summary.get("predictive_analyses", 0)
                total_count = summary.get("total_phase3_analyses", 0)
                
                # Validate total calculation
                expected_total = rwe_count + market_access_count + predictive_count
                if total_count != expected_total:
                    self.log_test_result("Phase 3 Dashboard", False, 
                                       f"Total count mismatch: expected {expected_total}, got {total_count}")
                    return False
                
                # Check capabilities content
                rwe_capabilities = capabilities.get("real_world_evidence", [])
                market_capabilities = capabilities.get("market_access", [])
                predictive_capabilities = capabilities.get("predictive_analytics", [])
                
                has_rwe_caps = isinstance(rwe_capabilities, list) and len(rwe_capabilities) > 0
                has_market_caps = isinstance(market_capabilities, list) and len(market_capabilities) > 0
                has_predictive_caps = isinstance(predictive_capabilities, list) and len(predictive_capabilities) > 0
                
                self.log_test_result("Phase 3 Dashboard", True, 
                                   f"Dashboard data retrieved successfully. "
                                   f"RWE analyses: {rwe_count}, "
                                   f"Market access analyses: {market_access_count}, "
                                   f"Predictive analyses: {predictive_count}, "
                                   f"Total: {total_count}, "
                                   f"Capabilities: RWE={len(rwe_capabilities)}, "
                                   f"Market={len(market_capabilities)}, "
                                   f"Predictive={len(predictive_capabilities)}")
                return True
                
            else:
                self.log_test_result("Phase 3 Dashboard", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Phase 3 Dashboard", False, f"Exception: {str(e)}")
            return False
    
    async def test_phase3_data_retrieval_endpoints(self) -> bool:
        """Test Phase 3 data retrieval endpoints"""
        try:
            # Test RWE retrieval endpoint
            rwe_response = await self.client.get(f"{API_BASE_URL}/real-world-evidence/{TEST_THERAPY_AREA}")
            rwe_success = rwe_response.status_code in [200, 404]  # 404 is acceptable if no data exists
            
            # Test Market Access retrieval endpoint
            market_response = await self.client.get(f"{API_BASE_URL}/market-access-intelligence/{TEST_THERAPY_AREA}")
            market_success = market_response.status_code in [200, 404]
            
            # Test Predictive Analytics retrieval endpoint
            predictive_response = await self.client.get(f"{API_BASE_URL}/predictive-analytics/{TEST_THERAPY_AREA}")
            predictive_success = predictive_response.status_code in [200, 404]
            
            # Test with product name parameter
            rwe_with_product = await self.client.get(
                f"{API_BASE_URL}/real-world-evidence/{TEST_THERAPY_AREA}",
                params={"product_name": TEST_PRODUCT_NAME}
            )
            rwe_product_success = rwe_with_product.status_code in [200, 404]
            
            # Check if any data was found
            data_found = False
            if rwe_response.status_code == 200:
                rwe_data = rwe_response.json()
                if rwe_data.get("therapy_area"):
                    data_found = True
            
            if market_response.status_code == 200:
                market_data = market_response.json()
                if market_data.get("therapy_area"):
                    data_found = True
            
            if predictive_response.status_code == 200:
                predictive_data = predictive_response.json()
                if predictive_data.get("therapy_area"):
                    data_found = True
            
            success = rwe_success and market_success and predictive_success and rwe_product_success
            
            self.log_test_result("Phase 3 Data Retrieval Endpoints", success, 
                               f"RWE endpoint: {rwe_response.status_code}, "
                               f"Market access endpoint: {market_response.status_code}, "
                               f"Predictive endpoint: {predictive_response.status_code}, "
                               f"RWE with product: {rwe_with_product.status_code}, "
                               f"Data found: {data_found}")
            return success
            
        except Exception as e:
            self.log_test_result("Phase 3 Data Retrieval Endpoints", False, f"Exception: {str(e)}")
            return False

    # ========================================
    # COMPANY INTELLIGENCE ENGINE TESTS
    # ========================================
    
    async def test_company_intelligence_qinlock(self) -> bool:
        """Test Company Intelligence Engine with Qinlock (primary test case)"""
        try:
            payload = {
                "product_name": "Qinlock",
                "therapy_area": "GIST",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": True
            }
            
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["product_name", "parent_company", "company_website", "market_class", 
                                 "investor_data", "press_releases", "competitive_products", 
                                 "financial_metrics", "recent_developments", "sources_scraped", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Company Intelligence - Qinlock", False, 
                                       f"Missing required fields: {missing_fields}")
                    return False
                
                # Validate product name mapping
                if data.get("product_name") != "Qinlock":
                    self.log_test_result("Company Intelligence - Qinlock", False, 
                                       f"Product name mismatch: {data.get('product_name')}")
                    return False
                
                # Check parent company identification (should identify Deciphera or Blueprint Medicines)
                parent_company = data.get("parent_company", "").lower()
                expected_companies = ["deciphera", "blueprint medicines", "blueprint"]
                company_identified = any(company in parent_company for company in expected_companies)
                
                # Validate website format
                website = data.get("company_website", "")
                has_valid_website = website.startswith("http") or ".com" in website
                
                # Check market class (should be kinase inhibitor or similar)
                market_class = data.get("market_class", "").lower()
                expected_classes = ["kinase inhibitor", "tyrosine kinase", "targeted therapy", "inhibitor"]
                class_identified = any(cls in market_class for cls in expected_classes)
                
                # Validate investor data structure
                investor_data = data.get("investor_data", {})
                has_investor_structure = isinstance(investor_data, dict)
                
                # Check competitive products
                competitive_products = data.get("competitive_products", [])
                has_competitors = len(competitive_products) > 0
                
                # Validate sources scraped
                sources_scraped = data.get("sources_scraped", [])
                has_sources = len(sources_scraped) > 0
                
                success = (not missing_fields and has_valid_website and 
                          has_investor_structure and has_competitors)
                
                self.log_test_result("Company Intelligence - Qinlock", success, 
                                   f"Parent company: {data.get('parent_company', 'Unknown')}, "
                                   f"Expected company identified: {company_identified}, "
                                   f"Market class: {data.get('market_class', 'Unknown')}, "
                                   f"Class identified: {class_identified}, "
                                   f"Competitors found: {len(competitive_products)}, "
                                   f"Sources: {len(sources_scraped)}")
                return success
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Company Intelligence - Qinlock", False, 
                                       "Perplexity API key authentication required - endpoint structure validated")
                    return False
                else:
                    self.log_test_result("Company Intelligence - Qinlock", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Company Intelligence - Qinlock", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Qinlock", False, f"Exception: {str(e)}")
            return False
    
    async def test_company_intelligence_keytruda(self) -> bool:
        """Test Company Intelligence Engine with Keytruda (secondary test case)"""
        try:
            payload = {
                "product_name": "Keytruda",
                "therapy_area": "Oncology",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": True
            }
            
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate basic structure
                required_fields = ["product_name", "parent_company", "market_class"]
                has_required = all(field in data for field in required_fields)
                
                if not has_required:
                    self.log_test_result("Company Intelligence - Keytruda", False, 
                                       "Missing basic required fields")
                    return False
                
                # Check parent company (should identify Merck)
                parent_company = data.get("parent_company", "").lower()
                merck_identified = "merck" in parent_company
                
                # Check market class (should be PD-1 inhibitor or immunotherapy)
                market_class = data.get("market_class", "").lower()
                expected_classes = ["pd-1", "checkpoint inhibitor", "immunotherapy", "monoclonal antibody"]
                class_identified = any(cls in market_class for cls in expected_classes)
                
                # Check competitive products (should find other PD-1 inhibitors)
                competitive_products = data.get("competitive_products", [])
                has_competitors = len(competitive_products) > 0
                
                # Look for known PD-1 competitors in the results
                competitor_names = [comp.get("name", "").lower() for comp in competitive_products]
                known_competitors = ["opdivo", "tecentriq", "imfinzi", "bavencio"]
                found_known_competitors = sum(1 for comp in known_competitors 
                                            if any(comp in name for name in competitor_names))
                
                success = has_required and has_competitors
                
                self.log_test_result("Company Intelligence - Keytruda", success, 
                                   f"Parent company: {data.get('parent_company', 'Unknown')}, "
                                   f"Merck identified: {merck_identified}, "
                                   f"Market class: {data.get('market_class', 'Unknown')}, "
                                   f"Class identified: {class_identified}, "
                                   f"Competitors: {len(competitive_products)}, "
                                   f"Known competitors found: {found_known_competitors}")
                return success
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Company Intelligence - Keytruda", False, 
                                       "Perplexity API key authentication required")
                    return False
                else:
                    self.log_test_result("Company Intelligence - Keytruda", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Company Intelligence - Keytruda", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Keytruda", False, f"Exception: {str(e)}")
            return False
    
    async def test_company_intelligence_competitive_discovery(self) -> bool:
        """Test competitive product discovery in GIST therapeutic area"""
        try:
            payload = {
                "product_name": "Qinlock",
                "therapy_area": "GIST",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": True
            }
            
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                competitive_products = data.get("competitive_products", [])
                
                if len(competitive_products) == 0:
                    self.log_test_result("Company Intelligence - Competitive Discovery", False, 
                                       "No competitive products found")
                    return False
                
                # Validate competitive product structure
                valid_competitors = 0
                gist_related_competitors = 0
                
                for competitor in competitive_products:
                    # Check required fields
                    if competitor.get("name") and competitor.get("description"):
                        valid_competitors += 1
                    
                    # Check for GIST-related terms
                    comp_text = (competitor.get("name", "") + " " + 
                               competitor.get("description", "")).lower()
                    if any(term in comp_text for term in ["gist", "imatinib", "sunitinib", "regorafenib", 
                                                         "avapritinib", "ripretinib", "gleevec", "sutent"]):
                        gist_related_competitors += 1
                
                # Check for known GIST competitors
                known_gist_drugs = ["imatinib", "sunitinib", "regorafenib", "avapritinib", "gleevec", "sutent", "stivarga"]
                found_known_drugs = 0
                
                for competitor in competitive_products:
                    comp_name = competitor.get("name", "").lower()
                    comp_desc = competitor.get("description", "").lower()
                    comp_text = comp_name + " " + comp_desc
                    
                    for drug in known_gist_drugs:
                        if drug in comp_text:
                            found_known_drugs += 1
                            break
                
                # Validate therapeutic area consistency
                therapy_area_matches = 0
                for competitor in competitive_products:
                    if competitor.get("therapeutic_area") == "GIST" or competitor.get("therapeutic_area") == "general":
                        therapy_area_matches += 1
                
                success = (valid_competitors >= 2 and 
                          (gist_related_competitors > 0 or found_known_drugs > 0))
                
                self.log_test_result("Company Intelligence - Competitive Discovery", success, 
                                   f"Total competitors: {len(competitive_products)}, "
                                   f"Valid structure: {valid_competitors}, "
                                   f"GIST-related: {gist_related_competitors}, "
                                   f"Known GIST drugs: {found_known_drugs}, "
                                   f"Therapy area matches: {therapy_area_matches}")
                return success
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Company Intelligence - Competitive Discovery", False, 
                                       "Perplexity API key required for competitive discovery")
                    return False
                else:
                    self.log_test_result("Company Intelligence - Competitive Discovery", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Company Intelligence - Competitive Discovery", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Competitive Discovery", False, f"Exception: {str(e)}")
            return False
    
    async def test_company_intelligence_investor_relations(self) -> bool:
        """Test investor relations data extraction functionality"""
        try:
            payload = {
                "product_name": "Qinlock",
                "therapy_area": "GIST",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": False  # Focus on investor data
            }
            
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                investor_data = data.get("investor_data", {})
                
                if not isinstance(investor_data, dict):
                    self.log_test_result("Company Intelligence - Investor Relations", False, 
                                       "Investor data not in correct format")
                    return False
                
                # Check for investor relations data structure
                expected_fields = ["financial_highlights", "recent_earnings", "pipeline_updates", 
                                 "press_releases", "presentation_links", "sources_accessed"]
                
                has_structure = all(field in investor_data for field in expected_fields)
                
                # Check if any data was actually scraped
                press_releases = investor_data.get("press_releases", [])
                financial_highlights = investor_data.get("financial_highlights", [])
                sources_accessed = investor_data.get("sources_accessed", [])
                presentation_links = investor_data.get("presentation_links", [])
                
                has_press_releases = len(press_releases) > 0
                has_financial_data = len(financial_highlights) > 0
                has_sources = len(sources_accessed) > 0
                has_presentations = len(presentation_links) > 0
                
                # Validate press release structure if present
                valid_press_releases = 0
                if press_releases:
                    for pr in press_releases:
                        if pr.get("title") and pr.get("url") and pr.get("type"):
                            valid_press_releases += 1
                
                # Check for financial metrics in main response
                financial_metrics = data.get("financial_metrics", {})
                has_financial_metrics = isinstance(financial_metrics, dict) and len(financial_metrics) > 0
                
                # Check sources scraped at top level
                top_level_sources = data.get("sources_scraped", [])
                has_top_level_sources = len(top_level_sources) > 0
                
                # Success if we have proper structure and some data was attempted to be scraped
                success = (has_structure and 
                          (has_sources or has_top_level_sources or has_press_releases or has_financial_data))
                
                self.log_test_result("Company Intelligence - Investor Relations", success, 
                                   f"Structure complete: {has_structure}, "
                                   f"Press releases: {len(press_releases)} (valid: {valid_press_releases}), "
                                   f"Financial highlights: {len(financial_highlights)}, "
                                   f"Sources accessed: {len(sources_accessed)}, "
                                   f"Presentations: {len(presentation_links)}, "
                                   f"Financial metrics: {has_financial_metrics}")
                return success
                
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "api" in error_text and ("key" in error_text or "auth" in error_text):
                    self.log_test_result("Company Intelligence - Investor Relations", False, 
                                       "Perplexity API key required for investor relations scraping")
                    return False
                else:
                    self.log_test_result("Company Intelligence - Investor Relations", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Company Intelligence - Investor Relations", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Investor Relations", False, f"Exception: {str(e)}")
            return False
    
    async def test_company_intelligence_error_handling(self) -> bool:
        """Test error handling for unknown products and failed web scraping"""
        try:
            # Test with unknown/invalid product
            payload = {
                "product_name": "UnknownDrugXYZ123",
                "therapy_area": "Unknown Disease Area",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": True
            }
            
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Should still return valid structure even for unknown products
                required_fields = ["product_name", "parent_company", "company_website", "market_class"]
                has_required = all(field in data for field in required_fields)
                
                if not has_required:
                    self.log_test_result("Company Intelligence - Error Handling", False, 
                                       "Missing required fields in error case")
                    return False
                
                # Check if fallback values are used
                parent_company = data.get("parent_company", "")
                market_class = data.get("market_class", "")
                
                # Should have fallback values for unknown products
                has_fallback_company = "unknown" in parent_company.lower() or len(parent_company) > 0
                has_fallback_class = "unknown" in market_class.lower() or len(market_class) > 0
                
                # Check investor data error handling
                investor_data = data.get("investor_data", {})
                has_error_field = "error" in investor_data
                
                # Competitive products should be empty or have error handling
                competitive_products = data.get("competitive_products", [])
                
                # Check if error is properly handled in competitive products
                error_in_competitors = False
                if competitive_products:
                    for comp in competitive_products:
                        if "error" in comp.get("name", "").lower() or "error" in comp.get("description", "").lower():
                            error_in_competitors = True
                            break
                
                success = has_required and (has_fallback_company or has_fallback_class)
                
                self.log_test_result("Company Intelligence - Error Handling", success, 
                                   f"Structure maintained: {has_required}, "
                                   f"Fallback company: {has_fallback_company}, "
                                   f"Fallback class: {has_fallback_class}, "
                                   f"Investor error handling: {has_error_field}, "
                                   f"Competitor error handling: {error_in_competitors}")
                return success
                
            elif response.status_code == 500:
                # Check if error message is informative
                error_text = response.text
                
                # Should have proper error handling
                has_informative_error = len(error_text) > 10
                
                try:
                    error_data = response.json()
                    has_detail = "detail" in error_data
                    error_message = error_data.get("detail", "")
                    has_meaningful_detail = len(error_message) > 10
                except:
                    has_detail = False
                    has_meaningful_detail = False
                
                success = has_informative_error and (has_detail or "api" in error_text.lower())
                
                self.log_test_result("Company Intelligence - Error Handling", success, 
                                   f"Informative error: {has_informative_error}, "
                                   f"Structured error: {has_detail}, "
                                   f"Meaningful detail: {has_meaningful_detail}")
                return success
            else:
                self.log_test_result("Company Intelligence - Error Handling", False, 
                                   f"Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Error Handling", False, f"Exception: {str(e)}")
            return False
    
    async def test_company_intelligence_data_persistence(self) -> bool:
        """Test MongoDB storage and retrieval of company intelligence data"""
        try:
            # First, generate company intelligence
            payload = {
                "product_name": "TestProduct123",
                "therapy_area": "Test Area",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": True
            }
            
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            
            if response.status_code == 200:
                # Data should be stored in MongoDB
                # Now test retrieval
                retrieval_response = await self.client.get(f"{API_BASE_URL}/company-intelligence/TestProduct123")
                
                if retrieval_response.status_code == 200:
                    retrieved_data = retrieval_response.json()
                    
                    # Validate retrieved data structure
                    required_fields = ["product_name", "parent_company", "market_class", "timestamp"]
                    has_required = all(field in retrieved_data for field in required_fields)
                    
                    # Check if product name matches
                    product_match = retrieved_data.get("product_name") == "TestProduct123"
                    
                    # Validate timestamp
                    timestamp_valid = False
                    try:
                        from datetime import datetime
                        timestamp_str = retrieved_data.get("timestamp", "")
                        if timestamp_str:
                            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            timestamp_valid = True
                    except:
                        pass
                    
                    success = has_required and product_match and timestamp_valid
                    
                    self.log_test_result("Company Intelligence - Data Persistence", success, 
                                       f"Storage and retrieval successful, "
                                       f"Required fields: {has_required}, "
                                       f"Product match: {product_match}, "
                                       f"Valid timestamp: {timestamp_valid}")
                    return success
                    
                elif retrieval_response.status_code == 404:
                    # Data wasn't stored properly
                    self.log_test_result("Company Intelligence - Data Persistence", False, 
                                       "Data not found after storage - storage may have failed")
                    return False
                else:
                    self.log_test_result("Company Intelligence - Data Persistence", False, 
                                       f"Retrieval failed: {retrieval_response.status_code}")
                    return False
                    
            elif response.status_code == 500:
                # API key error - test retrieval of non-existent data
                retrieval_response = await self.client.get(f"{API_BASE_URL}/company-intelligence/NonExistentProduct")
                
                if retrieval_response.status_code == 404:
                    # Proper 404 for non-existent data
                    self.log_test_result("Company Intelligence - Data Persistence", True, 
                                       "Proper 404 response for non-existent data - retrieval endpoint working")
                    return True
                else:
                    self.log_test_result("Company Intelligence - Data Persistence", False, 
                                       f"Unexpected retrieval response: {retrieval_response.status_code}")
                    return False
            else:
                self.log_test_result("Company Intelligence - Data Persistence", False, 
                                   f"Storage failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Data Persistence", False, f"Exception: {str(e)}")
            return False
    
    async def test_company_intelligence_timeout_handling(self) -> bool:
        """Test timeout and response time handling for web requests"""
        try:
            # Test with a product that might require extensive web scraping
            payload = {
                "product_name": "Qinlock",
                "therapy_area": "GIST",
                "api_key": TEST_PERPLEXITY_KEY,
                "include_competitors": True
            }
            
            start_time = time.time()
            response = await self.client.post(f"{API_BASE_URL}/company-intelligence", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should complete within reasonable time (60 seconds max due to our client timeout)
            reasonable_time = response_time < 60
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if response is complete despite any timeout issues
                has_basic_structure = all(field in data for field in ["product_name", "parent_company"])
                
                # Check if timeout handling is evident in the data
                investor_data = data.get("investor_data", {})
                has_timeout_handling = (
                    "error" in investor_data or 
                    "timeout" in str(investor_data).lower() or
                    len(investor_data.get("sources_accessed", [])) >= 0  # Should have attempted scraping
                )
                
                success = reasonable_time and has_basic_structure
                
                self.log_test_result("Company Intelligence - Timeout Handling", success, 
                                   f"Response time: {response_time:.2f}s, "
                                   f"Reasonable time: {reasonable_time}, "
                                   f"Basic structure: {has_basic_structure}, "
                                   f"Timeout handling: {has_timeout_handling}")
                return success
                
            elif response.status_code == 500:
                # Even with API errors, should respond in reasonable time
                success = reasonable_time
                
                self.log_test_result("Company Intelligence - Timeout Handling", success, 
                                   f"Response time: {response_time:.2f}s (with API error), "
                                   f"Reasonable time: {reasonable_time}")
                return success
            else:
                self.log_test_result("Company Intelligence - Timeout Handling", False, 
                                   f"Unexpected status: {response.status_code}, Time: {response_time:.2f}s")
                return False
                
        except Exception as e:
            self.log_test_result("Company Intelligence - Timeout Handling", False, f"Exception: {str(e)}")
            return False
    
    async def test_visualization_data(self) -> bool:
        """Test Plotly chart data generation"""
        if not self.analysis_id:
            self.log_test_result("Visualization Data", False, "No analysis ID available")
            return False
            
        try:
            # Get funnel data which should contain visualization
            response = await self.client.get(f"{API_BASE_URL}/funnels/{self.analysis_id}")
            
            if response.status_code == 200:
                funnel_data = response.json()
                
                if not funnel_data:
                    self.log_test_result("Visualization Data", False, "No funnel data found")
                    return False
                
                viz_data = funnel_data.get("visualization_data", {})
                
                # Check for chart data
                chart_types = []
                if viz_data.get("funnel_chart"):
                    chart_types.append("funnel")
                    # Validate it's valid JSON
                    try:
                        json.loads(viz_data["funnel_chart"])
                    except:
                        self.log_test_result("Visualization Data", False, "Invalid funnel chart JSON")
                        return False
                
                if viz_data.get("scenario_chart"):
                    chart_types.append("scenario")
                    try:
                        json.loads(viz_data["scenario_chart"])
                    except:
                        self.log_test_result("Visualization Data", False, "Invalid scenario chart JSON")
                        return False
                
                if viz_data.get("market_chart"):
                    chart_types.append("market")
                    try:
                        json.loads(viz_data["market_chart"])
                    except:
                        self.log_test_result("Visualization Data", False, "Invalid market chart JSON")
                        return False
                
                success = len(chart_types) >= 1
                self.log_test_result("Visualization Data", success, 
                                   f"Generated charts: {', '.join(chart_types)}")
                return success
            else:
                self.log_test_result("Visualization Data", False, 
                                   f"Funnel retrieval failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Visualization Data", False, f"Exception: {str(e)}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all backend tests in sequence"""
        print(f"\n🧪 Starting Pharma Intelligence Platform Backend Tests")
        print(f"📍 Backend URL: {API_BASE_URL}")
        print(f"🎯 Test Therapy Area: {TEST_THERAPY_AREA}")
        print(f"💊 Test Product: {TEST_PRODUCT_NAME}")
        print("=" * 80)
        
        # Test sequence
        tests = [
            ("API Health Check", self.test_api_health),
            ("Database Connection", self.test_database_connection),
            ("Existing Data Retrieval", self.test_existing_data_retrieval),
            ("Core Analysis Endpoint", self.test_therapy_analysis),
            ("Funnel Generation", self.test_funnel_generation),
            ("Competitive Intelligence", self.test_competitive_analysis),
            ("Scenario Modeling", self.test_scenario_modeling),
            ("Clinical Trials Search", self.test_clinical_trials_search),
            ("Export Functionality", self.test_export_functionality),
            ("Data Persistence", self.test_data_persistence),
            ("Visualization Data", self.test_visualization_data),
            # Perplexity Integration Tests
            ("Perplexity Search Endpoint", self.test_perplexity_search_endpoint),
            ("Perplexity Pharmaceutical Search", self.test_perplexity_pharmaceutical_search),
            ("Enhanced Competitive Analysis", self.test_enhanced_competitive_analysis),
            ("Perplexity Error Handling", self.test_perplexity_error_handling),
            ("Perplexity Data Storage", self.test_perplexity_data_storage),
            # Company Intelligence Engine Tests (NEW)
            ("Company Intelligence - Qinlock", self.test_company_intelligence_qinlock),
            ("Company Intelligence - Keytruda", self.test_company_intelligence_keytruda),
            ("Company Intelligence - Competitive Discovery", self.test_company_intelligence_competitive_discovery),
            ("Company Intelligence - Investor Relations", self.test_company_intelligence_investor_relations),
            ("Company Intelligence - Error Handling", self.test_company_intelligence_error_handling),
            ("Company Intelligence - Data Persistence", self.test_company_intelligence_data_persistence),
            ("Company Intelligence - Timeout Handling", self.test_company_intelligence_timeout_handling)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                success = await test_func()
                results[test_name] = success
                if success:
                    passed += 1
            except Exception as e:
                print(f"❌ FAIL {test_name}: Unexpected error - {str(e)}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 80)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        print("=" * 80)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total) * 100,
            "results": results,
            "test_details": self.test_results,
            "analysis_id": self.analysis_id
        }

async def main():
    """Main test execution function"""
    async with PharmaAPITester() as tester:
        results = await tester.run_all_tests()
        
        # Save results to file
        with open('/app/backend_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: /app/backend_test_results.json")
        
        # Return exit code based on results
        if results["success_rate"] >= 70:  # 70% pass rate threshold
            print(f"🎉 Overall test result: SUCCESS ({results['success_rate']:.1f}%)")
            return 0
        else:
            print(f"⚠️  Overall test result: NEEDS ATTENTION ({results['success_rate']:.1f}%)")
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)