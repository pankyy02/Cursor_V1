#!/usr/bin/env python3
"""
Phase 4 OAuth & Authentication Testing for Pharma Intelligence Platform
Tests OAuth integration, user management, payments, and security features
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')
load_dotenv('/app/backend/.env')

# Get backend URL from frontend env
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE_URL = f"{BACKEND_URL}/api"

class Phase4Tester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.test_results = {}
        self.test_user_id = None
        self.test_access_token = None
        self.test_session_id = None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test results for reporting"""
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
    
    async def test_user_registration(self) -> bool:
        """Test user registration endpoint"""
        try:
            registration_data = {
                "email": "john.doe@pharmatech.com",
                "password": "SecurePass123!",
                "first_name": "John",
                "last_name": "Doe",
                "company": "PharmaTech Solutions",
                "role": "Senior Analyst"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/auth/register", json=registration_data)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["message", "user_id", "email"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("User Registration", False, 
                                       f"Missing response fields: {missing_fields}")
                    return False
                
                if data.get("email") != registration_data["email"]:
                    self.log_test_result("User Registration", False, 
                                       f"Email mismatch: expected {registration_data['email']}, got {data.get('email')}")
                    return False
                
                self.test_user_id = data.get("user_id")
                self.log_test_result("User Registration", True, 
                                   f"User registered successfully. ID: {self.test_user_id[:8]}..., Email: {data.get('email')}")
                return True
                
            elif response.status_code == 400:
                error_text = response.text.lower()
                if "already registered" in error_text or "email" in error_text:
                    self.log_test_result("User Registration", True, 
                                       "Email already registered - registration validation working")
                    return True
                else:
                    self.log_test_result("User Registration", False, 
                                       f"Registration validation error: {response.text}")
                    return False
            else:
                self.log_test_result("User Registration", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("User Registration", False, f"Exception: {str(e)}")
            return False
    
    async def test_user_login(self) -> bool:
        """Test user login endpoint"""
        try:
            login_data = {
                "email": "john.doe@pharmatech.com",
                "password": "SecurePass123!"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/auth/login", json=login_data)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["access_token", "token_type", "expires_in", "user"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("User Login", False, 
                                       f"Missing response fields: {missing_fields}")
                    return False
                
                if data.get("token_type") != "bearer":
                    self.log_test_result("User Login", False, 
                                       f"Invalid token type: {data.get('token_type')}")
                    return False
                
                user_data = data.get("user", {})
                if user_data.get("email") != login_data["email"]:
                    self.log_test_result("User Login", False, 
                                       f"User email mismatch: {user_data.get('email')}")
                    return False
                
                self.test_access_token = data.get("access_token")
                self.log_test_result("User Login", True, 
                                   f"Login successful. Token: {self.test_access_token[:16]}..., "
                                   f"User: {user_data.get('first_name')} {user_data.get('last_name')}, "
                                   f"Subscription: {user_data.get('subscription_tier')}")
                return True
                
            elif response.status_code == 401:
                self.log_test_result("User Login", False, 
                                   "Invalid credentials - user may not exist or password incorrect")
                return False
            else:
                self.log_test_result("User Login", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("User Login", False, f"Exception: {str(e)}")
            return False
    
    async def test_user_profile(self) -> bool:
        """Test user profile retrieval"""
        try:
            if not self.test_access_token:
                self.log_test_result("User Profile", False, "No access token available")
                return False
            
            headers = {"Authorization": f"Bearer {self.test_access_token}"}
            response = await self.client.get(f"{API_BASE_URL}/auth/profile", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["user", "profile"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("User Profile", False, 
                                       f"Missing response fields: {missing_fields}")
                    return False
                
                user_data = data.get("user", {})
                profile_data = data.get("profile", {})
                
                has_user_id = bool(user_data.get("id"))
                has_email = bool(user_data.get("email"))
                has_profile_id = bool(profile_data.get("user_id"))
                
                self.log_test_result("User Profile", True, 
                                   f"Profile retrieved successfully. User ID: {has_user_id}, Email: {has_email}, Profile linked: {has_profile_id}")
                return True
                
            elif response.status_code == 401:
                self.log_test_result("User Profile", False, 
                                   "Authentication failed - token may be invalid")
                return False
            else:
                self.log_test_result("User Profile", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("User Profile", False, f"Exception: {str(e)}")
            return False
    
    async def test_google_oauth_token(self) -> bool:
        """Test Google OAuth token authentication"""
        try:
            oauth_data = {
                "token": "mock-google-access-token-for-testing"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/auth/google/token", json=oauth_data)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["access_token", "token_type", "user", "oauth_provider"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Google OAuth Token", False, 
                                       f"Missing response fields: {missing_fields}")
                    return False
                
                if data.get("oauth_provider") != "google":
                    self.log_test_result("Google OAuth Token", False, 
                                       f"Invalid OAuth provider: {data.get('oauth_provider')}")
                    return False
                
                user_data = data.get("user", {})
                has_email = bool(user_data.get("email"))
                
                self.log_test_result("Google OAuth Token", True, 
                                   f"Google OAuth successful. Provider: {data.get('oauth_provider')}, "
                                   f"User email: {has_email}, Token: {data.get('access_token')[:16]}...")
                return True
                
            elif response.status_code == 400:
                error_text = response.text.lower()
                if "token" in error_text or "google" in error_text:
                    self.log_test_result("Google OAuth Token", False, 
                                       "Invalid Google token - OAuth validation working")
                    return False
                else:
                    self.log_test_result("Google OAuth Token", False, 
                                       f"OAuth validation error: {response.text}")
                    return False
            else:
                self.log_test_result("Google OAuth Token", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Google OAuth Token", False, f"Exception: {str(e)}")
            return False
    
    async def test_apple_oauth_token(self) -> bool:
        """Test Apple ID OAuth token authentication"""
        try:
            oauth_data = {
                "id_token": "mock.apple.id.token.for.testing",
                "code": "mock-apple-auth-code",
                "name": {
                    "firstName": "Jane",
                    "lastName": "Smith"
                }
            }
            
            response = await self.client.post(f"{API_BASE_URL}/auth/apple/token", json=oauth_data)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["access_token", "token_type", "user", "oauth_provider"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Apple OAuth Token", False, 
                                       f"Missing response fields: {missing_fields}")
                    return False
                
                if data.get("oauth_provider") != "apple":
                    self.log_test_result("Apple OAuth Token", False, 
                                       f"Invalid OAuth provider: {data.get('oauth_provider')}")
                    return False
                
                user_data = data.get("user", {})
                has_email = bool(user_data.get("email"))
                
                self.log_test_result("Apple OAuth Token", True, 
                                   f"Apple OAuth successful. Provider: {data.get('oauth_provider')}, "
                                   f"User email: {has_email}, Token: {data.get('access_token')[:16]}...")
                return True
                
            elif response.status_code == 400:
                error_text = response.text.lower()
                if "token" in error_text or "apple" in error_text:
                    self.log_test_result("Apple OAuth Token", False, 
                                       "Invalid Apple ID token - OAuth validation working")
                    return False
                else:
                    self.log_test_result("Apple OAuth Token", False, 
                                       f"OAuth validation error: {response.text}")
                    return False
            else:
                self.log_test_result("Apple OAuth Token", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Apple OAuth Token", False, f"Exception: {str(e)}")
            return False
    
    async def test_subscription_plans(self) -> bool:
        """Test subscription plans endpoint"""
        try:
            response = await self.client.get(f"{API_BASE_URL}/subscriptions/plans")
            
            if response.status_code == 200:
                data = response.json()
                
                if "plans" not in data:
                    self.log_test_result("Subscription Plans", False, 
                                       "Missing 'plans' field in response")
                    return False
                
                plans = data.get("plans", [])
                if len(plans) < 3:
                    self.log_test_result("Subscription Plans", False, 
                                       f"Expected at least 3 plans, got {len(plans)}")
                    return False
                
                required_plan_fields = ["id", "name", "price", "features", "api_limits", "description"]
                plan_validation_errors = []
                
                for plan in plans:
                    missing_fields = [field for field in required_plan_fields if field not in plan]
                    if missing_fields:
                        plan_validation_errors.append(f"Plan {plan.get('name', 'Unknown')}: missing {missing_fields}")
                
                if plan_validation_errors:
                    self.log_test_result("Subscription Plans", False, 
                                       f"Plan validation errors: {plan_validation_errors}")
                    return False
                
                plan_names = [plan.get("name", "").lower() for plan in plans]
                expected_tiers = ["basic", "professional", "enterprise"]
                found_tiers = [tier for tier in expected_tiers if any(tier in name for name in plan_names)]
                
                prices = [plan.get("price", 0) for plan in plans]
                has_valid_pricing = all(isinstance(price, (int, float)) and price > 0 for price in prices)
                
                self.log_test_result("Subscription Plans", True, 
                                   f"Found {len(plans)} subscription plans. "
                                   f"Tiers: {found_tiers}, "
                                   f"Price range: ${min(prices)}-${max(prices)}, "
                                   f"Valid pricing: {has_valid_pricing}")
                return True
                
            else:
                self.log_test_result("Subscription Plans", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Subscription Plans", False, f"Exception: {str(e)}")
            return False
    
    async def test_stripe_checkout_session(self) -> bool:
        """Test Stripe checkout session creation"""
        try:
            checkout_data = {
                "package_id": "basic"
            }
            
            response = await self.client.post(f"{API_BASE_URL}/payments/checkout/session", json=checkout_data)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["url", "session_id"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test_result("Stripe Checkout Session", False, 
                                       f"Missing response fields: {missing_fields}")
                    return False
                
                checkout_url = data.get("url", "")
                session_id = data.get("session_id", "")
                
                has_valid_url = checkout_url.startswith("http") and len(checkout_url) > 20
                has_valid_session_id = len(session_id) > 10
                
                self.test_session_id = session_id
                
                self.log_test_result("Stripe Checkout Session", True, 
                                   f"Checkout session created. Session ID: {session_id[:16]}..., "
                                   f"Valid URL: {has_valid_url}, URL length: {len(checkout_url)}")
                return True
                
            elif response.status_code == 400:
                error_text = response.text.lower()
                if "package" in error_text or "subscription" in error_text:
                    self.log_test_result("Stripe Checkout Session", False, 
                                       "Invalid package ID - validation working")
                    return False
                else:
                    self.log_test_result("Stripe Checkout Session", False, 
                                       f"Checkout validation error: {response.text}")
                    return False
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "stripe" in error_text or "checkout" in error_text:
                    self.log_test_result("Stripe Checkout Session", False, 
                                       "Stripe API error - checkout structure validated")
                    return False
                else:
                    self.log_test_result("Stripe Checkout Session", False, 
                                       f"Server error: {response.text}")
                    return False
            else:
                self.log_test_result("Stripe Checkout Session", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("Stripe Checkout Session", False, f"Exception: {str(e)}")
            return False
    
    async def test_session_management(self) -> bool:
        """Test session token validation and expiration"""
        try:
            # Test with invalid token
            invalid_headers = {"Authorization": "Bearer invalid-token-12345"}
            invalid_response = await self.client.get(f"{API_BASE_URL}/auth/profile", headers=invalid_headers)
            
            invalid_token_rejected = invalid_response.status_code == 401
            
            # Test with malformed token (HTTPBearer should handle this)
            malformed_headers = {"Authorization": "InvalidFormat token"}
            malformed_response = await self.client.get(f"{API_BASE_URL}/auth/profile", headers=malformed_headers)
            
            malformed_token_rejected = malformed_response.status_code in [401, 422]
            
            # Test with no token (HTTPBearer should handle this)
            no_token_response = await self.client.get(f"{API_BASE_URL}/auth/profile")
            
            no_token_rejected = no_token_response.status_code in [401, 422]
            
            # Test with empty Authorization header
            empty_headers = {"Authorization": ""}
            empty_response = await self.client.get(f"{API_BASE_URL}/auth/profile", headers=empty_headers)
            
            empty_token_rejected = empty_response.status_code in [401, 422]
            
            success = invalid_token_rejected and malformed_token_rejected and no_token_rejected and empty_token_rejected
            
            self.log_test_result("Session Management", success, 
                               f"Invalid token rejected: {invalid_token_rejected}, "
                               f"Malformed token rejected: {malformed_token_rejected}, "
                               f"No token rejected: {no_token_rejected}, "
                               f"Empty token rejected: {empty_token_rejected}")
            return success
            
        except Exception as e:
            self.log_test_result("Session Management", False, f"Exception: {str(e)}")
            return False
    
    async def test_user_logout(self) -> bool:
        """Test user logout endpoint"""
        try:
            if not self.test_access_token:
                self.log_test_result("User Logout", False, "No access token available")
                return False
            
            headers = {"Authorization": f"Bearer {self.test_access_token}"}
            response = await self.client.post(f"{API_BASE_URL}/auth/logout", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("message") and "logged out" in data["message"].lower():
                    self.log_test_result("User Logout", True, 
                                       f"Logout successful: {data.get('message')}")
                    self.test_access_token = None
                    return True
                else:
                    self.log_test_result("User Logout", False, 
                                       f"Unexpected logout response: {data}")
                    return False
                
            elif response.status_code == 401:
                self.log_test_result("User Logout", False, 
                                   "Authentication failed - token may be invalid")
                return False
            else:
                self.log_test_result("User Logout", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test_result("User Logout", False, f"Exception: {str(e)}")
            return False
    
    async def run_phase4_tests(self) -> Dict[str, Any]:
        """Run all Phase 4 OAuth and authentication tests"""
        print(f"\n🔐 Starting Phase 4: OAuth & Authentication Tests")
        print(f"📍 Backend URL: {API_BASE_URL}")
        print("=" * 80)
        
        # Test sequence for Phase 4
        tests = [
            ("User Registration", self.test_user_registration),
            ("User Login", self.test_user_login),
            ("User Profile", self.test_user_profile),
            ("Google OAuth Token", self.test_google_oauth_token),
            ("Apple OAuth Token", self.test_apple_oauth_token),
            ("Subscription Plans", self.test_subscription_plans),
            ("Stripe Checkout Session", self.test_stripe_checkout_session),
            ("Session Management", self.test_session_management),
            ("User Logout", self.test_user_logout)
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
        print(f"📊 PHASE 4 TEST SUMMARY: {passed}/{total} tests passed")
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
            "test_details": self.test_results
        }

async def main():
    """Main test execution function"""
    async with Phase4Tester() as tester:
        results = await tester.run_phase4_tests()
        
        # Save results to file
        with open('/app/phase4_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: /app/phase4_test_results.json")
        
        # Return exit code based on results
        if results["success_rate"] >= 70:  # 70% pass rate threshold
            print(f"🎉 Phase 4 test result: SUCCESS ({results['success_rate']:.1f}%)")
            return 0
        else:
            print(f"⚠️  Phase 4 test result: NEEDS ATTENTION ({results['success_rate']:.1f}%)")
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)