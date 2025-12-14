"use client"

import type React from "react"

import { useState, useMemo, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { InfoIcon, Loader2Icon, TrendingUpIcon, BarChart3Icon, ZapIcon, ActivityIcon } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Slider } from "@/components/ui/slider"

interface OptionParams {
  S: number
  K: number
  T: number
  sigma: number
  r: number
  type: number
}

interface PricingResult {
  gbdt_price: number
  bs_price: number
  mc_price?: number
}

const PRESETS = {
  atmCall: {
    name: "At-the-money Call",
    params: { S: 100, K: 100, T: 1, sigma: 0.2, r: 0.05, type: 1 },
  },
  otmPut: {
    name: "Out-of-the-money Put",
    params: { S: 100, K: 90, T: 0.5, sigma: 0.25, r: 0.03, type: 0 },
  },
  itmCall: {
    name: "In-the-money Call",
    params: { S: 110, K: 100, T: 0.75, sigma: 0.3, r: 0.04, type: 1 },
  },
}

export function OptionPricingDashboard() {
  const [params, setParams] = useState<OptionParams>({
    S: 100,
    K: 100,
    T: 1,
    sigma: 0.2,
    r: 0.05,
    type: 1,
  })
  const [result, setResult] = useState<PricingResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [interactiveMode, setInteractiveMode] = useState(false)
  const [showVisualization, setShowVisualization] = useState(true)

  const isValid = params.S > 0 && params.K > 0 && params.T > 0 && params.sigma >= 0

  const metrics = useMemo(() => {
    const moneyness = params.S / params.K
    const intrinsicValue = params.type === 1 ? Math.max(params.S - params.K, 0) : Math.max(params.K - params.S, 0)
    const timeValue = result ? result.gbdt_price - intrinsicValue : 0

    return {
      moneyness,
      intrinsicValue,
      timeValue,
      moneynessLabel: moneyness > 1.05 ? "In-the-money" : moneyness < 0.95 ? "Out-of-the-money" : "At-the-money",
    }
  }, [params.S, params.K, params.type, result])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!isValid) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      })

      if (!response.ok) {
        throw new Error("Failed to fetch pricing data")
      }

      const data: PricingResult = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!interactiveMode || !isValid) return

    const timer = setTimeout(() => {
      handleSubmit({ preventDefault: () => {} } as React.FormEvent)
    }, 800)

    return () => clearTimeout(timer)
  }, [params, interactiveMode])

  const applyPreset = (preset: typeof PRESETS.atmCall) => {
    setParams(preset.params)
    if (!interactiveMode) {
      setResult(null)
    }
    setError(null)
  }

  const chartData = useMemo(() => {
    if (!result) return []
    return [
      { method: "GBDT", value: result.gbdt_price, color: "bg-blue-500" },
      { method: "Black-Scholes", value: result.bs_price, color: "bg-purple-500" },
      ...(result.mc_price ? [{ method: "Monte Carlo", value: result.mc_price, color: "bg-emerald-500" }] : []),
    ]
  }, [result])

  const maxValue = Math.max(...chartData.map((d) => d.value), 1)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50">
      {/* Header with gradient accent */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10 shadow-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-500/30">
                <TrendingUpIcon className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                  European Option Pricing
                </h1>
                <p className="text-sm text-slate-600">ML-powered Monte Carlo approximation</p>
              </div>
            </div>
            <Button
              variant={interactiveMode ? "default" : "outline"}
              onClick={() => setInteractiveMode(!interactiveMode)}
              className="gap-2"
            >
              <ZapIcon className="h-4 w-4" />
              {interactiveMode ? "Interactive Mode ON" : "Interactive Mode OFF"}
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Input Form with enhanced design */}
          <Card className="lg:col-span-1 shadow-lg border-slate-200 hover:shadow-xl transition-shadow duration-300">
            <CardHeader className="bg-gradient-to-br from-slate-50 to-white">
              <CardTitle className="flex items-center gap-2">
                <BarChart3Icon className="h-5 w-5 text-blue-600" />
                Option Parameters
              </CardTitle>
              <CardDescription>Configure your European option</CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="spotPrice" className="text-sm font-semibold">
                      Spot Price (S)
                    </Label>
                    <span className="text-sm font-mono font-bold text-blue-600">${params.S.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[params.S]}
                    onValueChange={(value) => setParams({ ...params, S: value[0] })}
                    min={50}
                    max={200}
                    step={0.5}
                    className="py-2"
                  />
                  <Input
                    id="spotPrice"
                    type="number"
                    step="0.01"
                    min="0.01"
                    value={params.S}
                    onChange={(e) => setParams({ ...params, S: Number.parseFloat(e.target.value) || 0 })}
                    className="font-mono"
                    required
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="strikePrice" className="text-sm font-semibold">
                      Strike Price (K)
                    </Label>
                    <span className="text-sm font-mono font-bold text-blue-600">${params.K.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[params.K]}
                    onValueChange={(value) => setParams({ ...params, K: value[0] })}
                    min={50}
                    max={200}
                    step={0.5}
                    className="py-2"
                  />
                  <Input
                    id="strikePrice"
                    type="number"
                    step="0.01"
                    min="0.01"
                    value={params.K}
                    onChange={(e) => setParams({ ...params, K: Number.parseFloat(e.target.value) || 0 })}
                    className="font-mono"
                    required
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="timeToMaturity" className="text-sm font-semibold">
                      Time to Maturity (T)
                    </Label>
                    <span className="text-sm font-mono font-bold text-blue-600">
                      {params.T.toFixed(2)} {params.T === 1 ? "year" : "years"}
                    </span>
                  </div>
                  <Slider
                    value={[params.T]}
                    onValueChange={(value) => setParams({ ...params, T: value[0] })}
                    min={0.1}
                    max={5}
                    step={0.1}
                    className="py-2"
                  />
                  <Input
                    id="timeToMaturity"
                    type="number"
                    step="0.01"
                    min="0.01"
                    value={params.T}
                    onChange={(e) => setParams({ ...params, T: Number.parseFloat(e.target.value) || 0 })}
                    className="font-mono"
                    required
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="volatility" className="text-sm font-semibold">
                      Volatility (σ)
                    </Label>
                    <span className="text-sm font-mono font-bold text-blue-600">
                      {(params.sigma * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Slider
                    value={[params.sigma]}
                    onValueChange={(value) => setParams({ ...params, sigma: value[0] })}
                    min={0.05}
                    max={1}
                    step={0.01}
                    className="py-2"
                  />
                  <Input
                    id="volatility"
                    type="number"
                    step="0.01"
                    min="0"
                    value={params.sigma}
                    onChange={(e) => setParams({ ...params, sigma: Number.parseFloat(e.target.value) || 0 })}
                    className="font-mono"
                    required
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="riskFreeRate" className="text-sm font-semibold">
                      Risk-Free Rate (r)
                    </Label>
                    <span className="text-sm font-mono font-bold text-blue-600">{(params.r * 100).toFixed(2)}%</span>
                  </div>
                  <Slider
                    value={[params.r]}
                    onValueChange={(value) => setParams({ ...params, r: value[0] })}
                    min={0}
                    max={0.2}
                    step={0.001}
                    className="py-2"
                  />
                  <Input
                    id="riskFreeRate"
                    type="number"
                    step="0.001"
                    value={params.r}
                    onChange={(e) => setParams({ ...params, r: Number.parseFloat(e.target.value) || 0 })}
                    className="font-mono"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="optionType" className="text-sm font-semibold">
                    Option Type
                  </Label>
                  <Select
                    value={params.type.toString()}
                    onValueChange={(value) => setParams({ ...params, type: Number.parseInt(value) })}
                  >
                    <SelectTrigger id="optionType" className="font-medium">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Call Option</SelectItem>
                      <SelectItem value="0">Put Option</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {!interactiveMode && (
                  <Button
                    type="submit"
                    className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-lg shadow-blue-500/30"
                    disabled={!isValid || loading}
                  >
                    {loading ? (
                      <>
                        <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                        Calculating...
                      </>
                    ) : (
                      <>
                        <ActivityIcon className="mr-2 h-4 w-4" />
                        Calculate Option Price
                      </>
                    )}
                  </Button>
                )}

                {interactiveMode && loading && (
                  <div className="flex items-center justify-center gap-2 text-sm text-blue-600 py-2">
                    <Loader2Icon className="h-4 w-4 animate-spin" />
                    <span>Auto-updating...</span>
                  </div>
                )}

                {error && (
                  <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700 animate-in fade-in slide-in-from-top-2">
                    {error}
                  </div>
                )}
              </form>

              <div className="mt-6 space-y-3">
                <Label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Quick Presets</Label>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(PRESETS).map(([key, preset]) => (
                    <Button
                      key={key}
                      variant="outline"
                      size="sm"
                      onClick={() => applyPreset(preset)}
                      type="button"
                      className="justify-start hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700 transition-all duration-200"
                    >
                      {preset.name}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results with enhanced visualization */}
          <div className="lg:col-span-2 space-y-6">
            {result ? (
              <>
                <Card className="border-blue-200 bg-gradient-to-br from-blue-50 via-white to-blue-50/50 shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-blue-900">
                      Predicted Option Price (GBDT)
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <InfoIcon className="h-4 w-4 text-blue-600 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs">
                              XGBoost (Gradient Boosted Decision Trees) model trained on Monte Carlo simulations
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </CardTitle>
                    <CardDescription>Machine learning prediction</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="text-6xl font-bold bg-gradient-to-r from-blue-600 via-blue-700 to-blue-600 bg-clip-text text-transparent animate-in zoom-in duration-700">
                      ${result.gbdt_price.toFixed(4)}
                    </div>
                    <div className="flex gap-4 text-sm">
                      <div className="flex-1 rounded-lg bg-white/70 p-3 border border-blue-200">
                        <div className="text-slate-600">Intrinsic Value</div>
                        <div className="text-lg font-bold text-slate-900">${metrics.intrinsicValue.toFixed(4)}</div>
                      </div>
                      <div className="flex-1 rounded-lg bg-white/70 p-3 border border-blue-200">
                        <div className="text-slate-600">Time Value</div>
                        <div className="text-lg font-bold text-slate-900">${metrics.timeValue.toFixed(4)}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {showVisualization && chartData.length > 0 && (
                  <Card className="shadow-lg border-slate-200 animate-in fade-in slide-in-from-bottom-6 duration-700">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                          <BarChart3Icon className="h-5 w-5 text-slate-700" />
                          Price Comparison
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setShowVisualization(false)}
                          className="text-xs"
                        >
                          Hide
                        </Button>
                      </CardTitle>
                      <CardDescription>Visual comparison of pricing methods</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                    {chartData.map((item, index) => {
  const value = item.value ?? 0; // fallback to 0 if undefined
  return (
    <div key={item.method} className="space-y-2 animate-in slide-in-from-left duration-500" style={{ animationDelay: `${index * 100}ms` }}>
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-slate-700">{item.method}</span>
        <span className="font-mono font-bold text-slate-900">${value.toFixed(4)}</span>
      </div>
      <div className="relative h-8 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`${item.color} absolute inset-y-0 left-0 flex items-center justify-end pr-3 rounded-full transition-all duration-1000`}
          style={{ width: `${(value / maxValue) * 100}%` }}
        >
          <span className="text-xs font-bold text-white drop-shadow">${value.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
})}

                    </CardContent>
                  </Card>
                )}

                {!showVisualization && (
                  <Button
                    variant="outline"
                    onClick={() => setShowVisualization(true)}
                    className="w-full animate-in fade-in duration-300"
                  >
                    Show Price Comparison Chart
                  </Button>
                )}

                <div className="grid gap-6 md:grid-cols-2">
                  <Card className="shadow-lg border-purple-200 hover:shadow-xl hover:border-purple-300 transition-all duration-300 animate-in fade-in slide-in-from-left duration-700">
                    <CardHeader className="bg-gradient-to-br from-purple-50 to-white">
                      <CardTitle className="flex items-center gap-2 text-lg text-purple-900">
                        Black-Scholes Price
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <InfoIcon className="h-4 w-4 text-purple-600 cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent>
                              <p className="max-w-xs">Analytical solution using the Black-Scholes formula</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </CardTitle>
                      <CardDescription>Analytical method</CardDescription>
                    </CardHeader>
                    <CardContent className="pt-6">
                    <div className="text-4xl font-bold text-purple-700">
  ${result.bs_price !== undefined ? result.bs_price.toFixed(4) : "N/A"}
</div>

                      <div className="mt-3 flex items-center gap-2">
                        <div className="h-1 flex-1 bg-gradient-to-r from-purple-200 to-purple-400 rounded-full" />
                      </div>
                      <div className="mt-3 text-sm text-slate-600">
                        vs GBDT:{" "}
                        <span className="font-semibold text-slate-900">
  {result.bs_price !== undefined
    ? `$${Math.abs(result.gbdt_price - result.bs_price).toFixed(4)}`
    : "N/A"}
</span>
<span className="text-xs ml-1">
  {result.bs_price !== undefined
    ? `(${((Math.abs(result.gbdt_price - result.bs_price) / result.bs_price) * 100).toFixed(2)}%)`
    : ""}
</span>

                      </div>
                    </CardContent>
                  </Card>

                  {result.mc_price !== undefined && (
                    <Card className="shadow-lg border-emerald-200 hover:shadow-xl hover:border-emerald-300 transition-all duration-300 animate-in fade-in slide-in-from-right duration-700">
                      <CardHeader className="bg-gradient-to-br from-emerald-50 to-white">
                        <CardTitle className="flex items-center gap-2 text-lg text-emerald-900">
                          Monte Carlo Reference
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <InfoIcon className="h-4 w-4 text-emerald-600 cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">Simulated price using Monte Carlo methods</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </CardTitle>
                        <CardDescription>Simulation method</CardDescription>
                      </CardHeader>
                      <CardContent className="pt-6">
                        <div className="text-4xl font-bold text-emerald-700">${result.mc_price.toFixed(4)}</div>
                        <div className="mt-3 flex items-center gap-2">
                          <div className="h-1 flex-1 bg-gradient-to-r from-emerald-200 to-emerald-400 rounded-full" />
                        </div>
                        <div className="mt-3 text-sm text-slate-600">
                          vs GBDT:{" "}
                          <span className="font-semibold text-slate-900">
                            ${Math.abs(result.gbdt_price - result.mc_price).toFixed(4)}
                          </span>
                          <span className="text-xs ml-1">
                            ({((Math.abs(result.gbdt_price - result.mc_price) / result.mc_price) * 100).toFixed(2)}
                            %)
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>

                <Card className="shadow-lg border-slate-200 animate-in fade-in slide-in-from-bottom-8 duration-900">
                  <CardHeader className="bg-gradient-to-r from-slate-50 to-white">
                    <CardTitle className="text-lg">Option Details</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <dl className="grid grid-cols-2 gap-6">
                      <div className="space-y-1">
                        <dt className="text-xs font-medium text-slate-500 uppercase tracking-wide">Type</dt>
                        <dd className="flex items-center gap-2">
                          <span
                            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${
                              params.type === 1 ? "bg-green-100 text-green-800" : "bg-orange-100 text-orange-800"
                            }`}
                          >
                            {params.type === 1 ? "Call Option" : "Put Option"}
                          </span>
                        </dd>
                      </div>
                      <div className="space-y-1">
                        <dt className="text-xs font-medium text-slate-500 uppercase tracking-wide">Moneyness</dt>
                        <dd className="flex items-center gap-2">
                          <span
                            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${
                              metrics.moneyness > 1.05
                                ? "bg-blue-100 text-blue-800"
                                : metrics.moneyness < 0.95
                                  ? "bg-red-100 text-red-800"
                                  : "bg-amber-100 text-amber-800"
                            }`}
                          >
                            {metrics.moneynessLabel}
                          </span>
                        </dd>
                      </div>
                      <div className="space-y-1">
                        <dt className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                          Spot/Strike Ratio
                        </dt>
                        <dd className="text-lg font-bold font-mono text-slate-900">{metrics.moneyness.toFixed(4)}</dd>
                      </div>
                      <div className="space-y-1">
                        <dt className="text-xs font-medium text-slate-500 uppercase tracking-wide">Annualized Vol</dt>
                        <dd className="text-lg font-bold font-mono text-slate-900">
                          {(params.sigma * 100).toFixed(2)}%
                        </dd>
                      </div>
                      <div className="space-y-1">
                        <dt className="text-xs font-medium text-slate-500 uppercase tracking-wide">Days to Expiry</dt>
                        <dd className="text-lg font-bold font-mono text-slate-900">
                          {Math.round(params.T * 365)} days
                        </dd>
                      </div>
                      <div className="space-y-1">
                        <dt className="text-xs font-medium text-slate-500 uppercase tracking-wide">Risk-Free Rate</dt>
                        <dd className="text-lg font-bold font-mono text-slate-900">{(params.r * 100).toFixed(3)}%</dd>
                      </div>
                    </dl>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="lg:col-span-1 shadow-lg border-slate-200">
                <CardContent className="flex flex-col items-center justify-center py-20">
                  <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-100 to-blue-200 mb-6 animate-pulse">
                    <TrendingUpIcon className="h-10 w-10 text-blue-600" />
                  </div>
                  <h3 className="text-xl font-bold mb-2 text-slate-900">Ready to Calculate</h3>
                  <p className="text-sm text-slate-600 text-center max-w-sm mb-4">
                    {interactiveMode
                      ? "Adjust parameters using the sliders to see real-time pricing updates"
                      : 'Enter your option parameters and click "Calculate Option Price" to see results'}
                  </p>
                  {interactiveMode && (
                    <div className="flex items-center gap-2 text-xs text-blue-600 bg-blue-50 px-3 py-2 rounded-full">
                      <ZapIcon className="h-3 w-3" />
                      <span className="font-medium">Interactive mode enabled</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
