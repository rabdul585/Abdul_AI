import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

def test_serp_api():
    api_key = os.getenv("SERPAPI_API_KEY")
    print(f"Checking SERPAPI_API_KEY...")
    
    if not api_key:
        print("❌ SERPAPI_API_KEY not found in .env file.")
        return

    # Print first few chars to verify it's loaded (masking the rest)
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    print(f"ℹ️  Loaded Key: {masked_key}")

    print("\nAttempting a test search for 'Flexential data center'...")
    
    try:
        params = {
            "q": "Flexential data center",
            "api_key": api_key,
            "num": 1
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            print(f"❌ SerpApi Error: {results['error']}")
        elif "organic_results" in results:
            print("✅ SerpApi Connection Successful!")
            print(f"   Result: {results['organic_results'][0].get('title')}")
        else:
            print("⚠️  Connection successful but no organic results found.")
            print(f"   Raw keys returned: {results.keys()}")

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_serp_api()
