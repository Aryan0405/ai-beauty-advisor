import { useMemo, useState } from 'react'
import axios from 'axios'
import {
  AlertCircle,
  ChevronRight,
  LoaderCircle,
  Search,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react'
import './App.css'

const API_URL =
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000/api/v1'

const CATEGORIES = [
  'Cleanser',
  'Moisturizer',
  'Serum',
  'Mask',
  'Eye cream',
  'Sun protect',
  'Treatment',
]
const SKIN_TYPES = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']

function formatPrice(price) {
  if (price === null || price === undefined) return 'Price unavailable'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(price)
}

function App() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('')
  const [skinType, setSkinType] = useState('')
  const [minPrice, setMinPrice] = useState('')
  const [maxPrice, setMaxPrice] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)

  const visibleRecommendations = useMemo(() => {
    if (!result) return []

    return result.recommendations.filter((product) => {
      const price = Number(product.price)
      const meetsMinimum = !minPrice || price >= Number(minPrice)
      const meetsMaximum = !maxPrice || price <= Number(maxPrice)
      return meetsMinimum && meetsMaximum
    })
  }, [result, minPrice, maxPrice])

  async function handleSubmit(event) {
    event.preventDefault()
    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setError('Tell us what you are looking for before searching.')
      return
    }

    const filters = Object.fromEntries(
      Object.entries({ category, skin_type: skinType }).filter(([, value]) => value),
    )

    setLoading(true)
    setError('')
    setHasSearched(true)
    try {
      const response = await axios.post(`${API_URL}/recommend`, {
        query: trimmedQuery,
        filters: Object.keys(filters).length ? filters : undefined,
        top_k: 5,
      })
      setResult(response.data)
    } catch (requestError) {
      const message = requestError.response?.data?.detail
      setError(
        typeof message === 'string'
          ? message
          : 'We could not reach the beauty advisor. Check that the backend is running and try again.',
      )
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-panel" aria-labelledby="page-title">
        <div className="eyebrow"><Sparkles size={16} /> AI skincare matchmaker</div>
        <h1 id="page-title">Find products your skin will love.</h1>
        <p className="hero-copy">
          Describe your skin goals in your own words. We will match you with products and explain why they fit.
        </p>

        <form className="search-form" onSubmit={handleSubmit}>
          <label className="query-field">
            <Search size={21} aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. lightweight moisturizer for oily skin"
              aria-label="Describe your skincare needs"
            />
          </label>

          <div className="filters" aria-label="Search filters">
            <div className="filter-heading"><SlidersHorizontal size={16} /> Refine results</div>
            <label>
              Category
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                <option value="">Any category</option>
                {CATEGORIES.map((option) => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label>
              Skin type
              <select value={skinType} onChange={(event) => setSkinType(event.target.value)}>
                <option value="">Any skin type</option>
                {SKIN_TYPES.map((option) => <option key={option}>{option}</option>)}
              </select>
            </label>
            <label>
              Min. price
              <input type="number" min="0" value={minPrice} onChange={(event) => setMinPrice(event.target.value)} placeholder="$0" />
            </label>
            <label>
              Max. price
              <input type="number" min="0" value={maxPrice} onChange={(event) => setMaxPrice(event.target.value)} placeholder="Any" />
            </label>
          </div>

          <button className="search-button" type="submit" disabled={loading}>
            {loading ? <LoaderCircle className="spin" size={19} /> : <Search size={19} />}
            {loading ? 'Finding matches…' : 'Find my matches'}
          </button>
        </form>
      </section>

      <section className="results-panel" aria-live="polite">
        {error && (
          <div className="error-card" role="alert">
            <AlertCircle size={21} />
            <div><strong>Search unavailable</strong><span>{error}</span></div>
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <LoaderCircle className="spin" size={30} />
            <p>Looking through skincare matches and preparing your recommendations…</p>
          </div>
        )}

        {!loading && result && (
          <>
            <div className="results-heading">
              <div>
                <span className="section-kicker">Your personalized edit</span>
                <h2>{visibleRecommendations.length} product{visibleRecommendations.length === 1 ? '' : 's'} to consider</h2>
              </div>
              <span className="query-pill">“{result.query}”</span>
            </div>

            {visibleRecommendations.length > 0 ? (
              <div className="product-grid">
                {visibleRecommendations.map((product) => (
                  <article className="product-card" key={product.product_id}>
                    <div className="card-topline">
                      <span className="category-tag">{product.category}</span>
                      <span className="score">{Math.round(product.match_score * 100)}% match</span>
                    </div>
                    <p className="brand">{product.brand}</p>
                    <h3>{product.name}</h3>
                    <p className="ingredients">{product.ingredients}</p>
                    {product.explanation && (
                      <p className="product-explanation">
                        <Sparkles size={14} aria-hidden="true" />
                        {product.explanation}
                      </p>
                    )}
                    <div className="card-footer">
                      <strong>{formatPrice(product.price)}</strong>
                      <span>View details <ChevronRight size={16} /></span>
                    </div>
                  </article>
                ))}
              </div>
            ) : result.recommendations.length === 0 ? (
              <div className="empty-state">
                <h3>No products matched that search.</h3>
                <p>Try a different description, or loosen your category or skin type filters.</p>
              </div>
            ) : (
              <div className="empty-state">
                <h3>No products match that price range.</h3>
                <p>Try widening your price filters, or clear them and search again.</p>
              </div>
            )}
          </>
        )}

        {!loading && !result && !error && !hasSearched && (
          <div className="empty-state initial-state">
            <Sparkles size={30} />
            <h2>Your tailored routine starts here.</h2>
            <p>Try “gentle cleanser for sensitive skin” or “brightening serum under $50”.</p>
          </div>
        )}
      </section>
    </main>
  )
}

export default App
