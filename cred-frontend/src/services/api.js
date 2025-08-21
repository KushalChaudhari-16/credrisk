import axios from "axios"

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000"
// Remove the YAHOO_FINANCE_BASE_URL since we're using the proxy now

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error.response?.data || error.message)
    return Promise.reject(error)
  },
)

const api = {
  getCompanies: () => {
    return apiClient.get("/api/companies")
  },
  
  getCompanyScore: (symbol) => {
    return apiClient.get(`/api/companies/${symbol}/score`)
  },
  
  getCompanyHistory: (symbol) => {
    return apiClient.get(`/api/companies/${symbol}/history`)
  },
  
  getCompanyComparison: (symbol, agency = "sp") => {
    return apiClient.get(`/api/companies/${symbol}/compare/${agency}`)
  },
  
  getAlerts: () => {
    return apiClient.get("/api/alerts")
  },
  
  getMarketOverview: () => {
    return apiClient.get("/api/analytics/overview")
  },
  
  // Updated to use your backend proxy instead of direct Yahoo Finance call
  searchCompanies: async (query) => {
    try {
      const response = await apiClient.get("/api/search/companies", {
        params: { q: query },
      })
      return response
    } catch (error) {
      console.error("Company Search Error:", error)
      return { 
        data: { 
          success: false,
          quotes: [],
          error: error.message
        } 
      }
    }
  },
  
  healthCheck: () => {
    return apiClient.get("/api/system/health") // Updated to match your backend endpoint
  },
}

export default api