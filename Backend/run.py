
# updated_run.py
import os
import sys
import uvicorn

def main():
    print("=" * 60)
    print("🚀 CredTech Advanced API v2.0 Starting...")
    print("=" * 60)
    print("🎯 Advanced Multi-Model Credit Intelligence Platform")
    print("📊 Features: Multi-Model ML | Advanced Explainability | Real-time MLOps")
    print()
    print("🔗 API Endpoints:")
    print("   📖 Documentation: http://localhost:8000/docs")
    print("   ❤️  Health Check: http://localhost:8000/api/system/health")
    print("   🏢 Companies: http://localhost:8000/api/companies")
    print("   📈 Score Analysis: http://localhost:8000/api/companies/AAPL/score")
    print("   📊 Score History: http://localhost:8000/api/companies/AAPL/history")
    print("   ⚖️  Agency Compare: http://localhost:8000/api/companies/AAPL/compare/sp")
    print("   🚨 Active Alerts: http://localhost:8000/api/alerts")
    print("   📋 Analytics: http://localhost:8000/api/analytics/overview")
    print("=" * 60)
    
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()