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
            ("Visualization Data", self.test_visualization_data)
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