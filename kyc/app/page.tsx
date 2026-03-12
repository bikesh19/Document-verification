"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";

interface KYCData {
  name: string;
  dob: string;
  citizenship_no: string;
  dl_no: string;
  category: string;
}

interface VerificationResult {
  extracted_data: {
    name?: string;
    date_of_birth?: string;
    citizenship_number?: string;
    dl_number?: string;
    category?: string;
    blood_group?: string;
  };
  verification_status: string;
}

type VerificationStep =
  | "initializing"
  | "classification"
  | "ocr_text_extraction"
  | "field_extracting_parsing"
  | "verifying"
  | "completed";

const STEPS: { id: VerificationStep; label: string }[] = [
  { id: "initializing", label: "Initializing System" },
  { id: "classification", label: "Classification" },
  { id: "ocr_text_extraction", label: "OCR Text Extraction" },
  { id: "field_extracting_parsing", label: "Field Extracting & Parsing" },
  { id: "verifying", label: "Verifying" },
];

export default function Home() {
  const [formData, setFormData] = useState<KYCData>({
    name: "",
    dob: "",
    citizenship_no: "",
    dl_no: "",
    category: "",
  });

  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<VerificationStep | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [mismatches, setMismatches] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const compareData = (extracted: VerificationResult["extracted_data"]) => {
    const newMismatches: Record<string, boolean> = {};
    const normalize = (val: string | undefined) =>
      val?.trim() || "";

    const userFields = {
      name: formData.name,
      dob: formData.dob,
      citizenship_no: formData.citizenship_no,
      dl_no: formData.dl_no,
      category: formData.category,
    };

    const extractedFields = {
      name: extracted.name,
      dob: extracted.date_of_birth,
      citizenship_no: extracted.citizenship_number,
      dl_no: extracted.dl_number,
      category: extracted.category,
    };

    Object.keys(userFields).forEach((key) => {
      const userVal = normalize((userFields as any)[key]);
      const extVal = normalize((extractedFields as any)[key]);

      if (extVal && userVal !== extVal) {
        newMismatches[key] = true;
      } else if (!extVal && userVal) {
        newMismatches[key] = true;
      }
    });

    setMismatches(newMismatches);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!image) {
      alert("Please upload a license photo");
      return;
    }

    setLoading(true);
    setResult(null);
    setMismatches({});
    setError(null);
    setCurrentStep("initializing");

    const submitData = new FormData();
    submitData.append("image", image);

    try {
      const response = await fetch("/api/verify", {
        method: "POST",
        body: submitData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || "Verification request failed");
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader available");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const data = JSON.parse(line);

          if (data.status) {
            setCurrentStep(data.status);
          } else if (data.result) {
            setResult(data.result);
            if (data.result.extracted_data) {
              compareData(data.result.extracted_data);
            }
            setCurrentStep("completed");
          } else if (data.error) {
            throw new Error(data.error);
          }
        }
      }
    } catch (error: any) {
      console.error(error);
      setError(error.message || "An error occurred during verification");
      setCurrentStep(null);
    } finally {
      setLoading(false);
    }
  };

  const getStepStatus = (stepId: VerificationStep) => {
    if (!currentStep) return "pending";
    if (currentStep === "completed") return "completed";

    const currentIndex = STEPS.findIndex(s => s.id === currentStep);
    const stepIndex = STEPS.findIndex(s => s.id === stepId);

    if (stepIndex < currentIndex) return "completed";
    if (stepIndex === currentIndex) return "active";
    return "pending";
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent mb-2">
            KYC Verification
          </h1>
          <p className="text-slate-400">Nepal Driving License Smart Verification System</p>
        </header>

        {/* Global Error Banner */}
        {error && (
          <div className="mb-8 bg-red-500/10 border border-red-500/50 p-4 rounded-xl flex items-start gap-3 animate-in fade-in slide-in-from-top-4 duration-300">
            <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-sm font-semibold text-red-400">Connection Error</p>
              <p className="text-xs text-red-300 opacity-80 leading-relaxed">{error}</p>
            </div>
            <button onClick={() => setError(null)} className="ml-auto text-red-400/50 hover:text-red-400 transition-colors">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l18 18" /></svg>
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          {/* Form Section */}
          <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl backdrop-blur-sm">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <span className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm">1</span>
              User Details
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Full Name</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className={`w-full bg-slate-800 border ${mismatches.name ? 'border-red-500/50' : 'border-slate-700'} rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all`}
                  placeholder="John Doe"
                  required
                />
                {mismatches.name && <p className="text-xs text-red-400 mt-1">Found mismatch with document</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Date of Birth</label>
                <input
                  type="text"
                  name="dob"
                  value={formData.dob}
                  onChange={handleInputChange}
                  className={`w-full bg-slate-800 border ${mismatches.dob ? 'border-red-500/50' : 'border-slate-700'} rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all`}
                  placeholder="DD-MM-YYYY"
                  required
                />
                {mismatches.dob && <p className="text-xs text-red-400 mt-1">Mismatched</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Citizenship Number</label>
                <input
                  type="text"
                  name="citizenship_no"
                  value={formData.citizenship_no}
                  onChange={handleInputChange}
                  className={`w-full bg-slate-800 border ${mismatches.citizenship_no ? 'border-red-500/50' : 'border-slate-700'} rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all`}
                  placeholder="1234/5678"
                  required
                />
                {mismatches.citizenship_no && <p className="text-xs text-red-400 mt-1">Mismatched</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">DL Number</label>
                <input
                  type="text"
                  name="dl_no"
                  value={formData.dl_no}
                  onChange={handleInputChange}
                  className={`w-full bg-slate-800 border ${mismatches.dl_no ? 'border-red-500/50' : 'border-slate-700'} rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all`}
                  placeholder="01-06-00000000"
                  required
                />
                {mismatches.dl_no && <p className="text-xs text-red-400 mt-1">Mismatched</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Category</label>
                <input
                  type="text"
                  name="category"
                  value={formData.category}
                  onChange={handleInputChange}
                  className={`w-full bg-slate-800 border ${mismatches.category ? 'border-red-500/50' : 'border-slate-700'} rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all`}
                  placeholder="A, B"
                  required
                />
                {mismatches.category && <p className="text-xs text-red-400 mt-1">Mismatched</p>}
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded-xl transition-all shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed mt-4"
              >
                {loading ? "Processing..." : "Verify Identity"}
              </button>
            </form>
          </div>

          {/* Upload & Progress Section */}
          <div className="space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl backdrop-blur-sm">
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                <span className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm">2</span>
                Upload License
              </h2>

              <div
                onClick={() => !loading && fileInputRef.current?.click()}
                className={`border-2 border-dashed border-slate-700 rounded-xl p-8 text-center hover:border-blue-500 transition-colors ${loading ? 'cursor-not-allowed' : 'cursor-pointer'} group`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageChange}
                  className="hidden"
                  accept="image/*"
                  disabled={loading}
                />
                {!imagePreview ? (
                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mx-auto group-hover:bg-blue-500/20 transition-colors">
                      <svg className="w-6 h-6 text-slate-400 group-hover:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                      </svg>
                    </div>
                    <p className="text-slate-400 font-medium">Click to upload photo</p>
                    <p className="text-xs text-slate-500">JPG, PNG up to 5MB</p>
                  </div>
                ) : (
                  <div className="relative aspect-video rounded-lg overflow-hidden border border-slate-700 shadow-2xl">
                    <img src={imagePreview} alt="License Preview" className="object-cover w-full h-full" />
                    {!loading && (
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <p className="text-white text-sm">Change Image</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Progress Visualization */}
            {loading && (
              <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl backdrop-blur-sm animate-in fade-in zoom-in-95 duration-300">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6">Verification Progress</h3>
                <div className="space-y-4">
                  {STEPS.map((step, index) => {
                    const status = getStepStatus(step.id);
                    return (
                      <div key={step.id} className="flex items-center gap-4">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs transition-colors duration-300 ${status === 'completed' ? 'bg-green-500 text-white' :
                          status === 'active' ? 'bg-blue-500 text-white animate-pulse' :
                            'bg-slate-800 text-slate-500'
                          }`}>
                          {status === 'completed' ? (
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7" /></svg>
                          ) : index + 1}
                        </div>
                        <span className={`text-sm font-medium transition-colors duration-300 ${status === 'pending' ? 'text-slate-600' : 'text-slate-200'
                          }`}>
                          {step.label}
                        </span>
                        {status === 'active' && (
                          <div className="ml-auto w-4 h-4 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Results Section */}
            {result && !loading && (
              <div className={`bg-slate-900/50 border ${result.verification_status !== 'VERIFIED' || Object.keys(mismatches).length > 0
                ? 'border-yellow-500/50' : 'border-green-500/50'
                } p-6 rounded-2xl backdrop-blur-sm animate-in fade-in slide-in-from-bottom-4 duration-500`}>
                <h2 className="text-xl font-semibold mb-4">Verification Result</h2>
                <div className="space-y-4">
                  <div className="flex items-center justify-between py-2 border-b border-slate-800">
                    <span className="text-slate-400">System Status</span>
                    <span className={`font-bold ${result.verification_status === 'VERIFIED' ? 'text-green-400' :
                      result.verification_status === 'REJECTED' ? 'text-red-400' : 'text-yellow-400'
                      }`}>
                      {result.verification_status}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-slate-800">
                    <span className="text-slate-400">Data Comparison</span>
                    <span className={`font-bold ${!['VERIFIED', 'EXPIRED'].includes(result.verification_status) ? 'text-slate-500' :
                      Object.keys(mismatches).length === 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                      {!['VERIFIED', 'EXPIRED'].includes(result.verification_status)
                        ? "N/A - Verification Failed"
                        : Object.keys(mismatches).length === 0 ? "✓ PERFECT MATCH" : `⚠️ ${Object.keys(mismatches).length} MISMATCHES`
                      }
                    </span>
                  </div>

                  {result.verification_status === 'REJECTED' && (
                    <div className="bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                      <p className="text-xs text-red-400 leading-relaxed">
                        The uploaded document was not recognized as a valid Nepal Driving License. Data comparison was skipped for security.
                      </p>
                    </div>
                  )}

                  {result.verification_status === 'EXPIRED' && (
                    <div className="bg-orange-500/10 p-3 rounded-lg border border-orange-500/20">
                      <p className="text-xs text-orange-400 leading-relaxed font-semibold">
                        ⚠️ LICENSE EXPIRED: This document is recognized but has expired. Data comparison is provided for identity verification only.
                      </p>
                    </div>
                  )}

                  {['VERIFIED', 'EXPIRED'].includes(result.verification_status) && Object.keys(mismatches).length > 0 && (
                    <div className="bg-yellow-500/10 p-3 rounded-lg border border-yellow-500/20">
                      <p className="text-xs text-yellow-400 leading-relaxed">
                        Some fields provided do not match the data extracted from the document. Please verify your inputs.
                      </p>
                    </div>
                  )}

                  {['VERIFIED', 'EXPIRED'].includes(result.verification_status) && Object.keys(mismatches).length === 0 && (
                    <div className="bg-green-500/10 p-3 rounded-lg border border-green-500/20">
                      <p className="text-xs text-green-400 leading-relaxed">
                        All user-provided data matches the information extracted from the document perfectly.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
