import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 1 minute timeout for OCR

    try {
        const formData = await request.formData();
        const image = formData.get("image") as File;

        if (!image) {
            return NextResponse.json({ error: "No image provided" }, { status: 400 });
        }

        const flaskFormData = new FormData();
        flaskFormData.append("image", image);

        console.log("Forwarding request to Flask backend...");

        const response = await fetch("http://127.0.0.1:5000/verify", {
            method: "POST",
            body: flaskFormData,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json(
                { error: `Backend error: ${errorText}` },
                { status: response.status }
            );
        }

        const stream = response.body;
        return new Response(stream, {
            headers: {
                "Content-Type": "application/x-ndjson",
            },
        });
    } catch (error: any) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
            return NextResponse.json(
                { error: "Verification processing timed out. The image might be too complex or the server is busy." },
                { status: 504 }
            );
        }

        console.error("Verification error:", error);

        // Handle common connection errors
        const message = error.message || "";
        if (message.includes("fetch failed") || message.includes("ECONNREFUSED")) {
            return NextResponse.json(
                { error: "Backend Server Unavailable. Please ensure 'verify_api.py' is running on port 5000." },
                { status: 503 }
            );
        }

        return NextResponse.json(
            { error: `Verification failed: ${error.message}` },
            { status: 500 }
        );
    }
}
