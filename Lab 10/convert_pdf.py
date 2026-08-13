import os
import subprocess

html_path = r"d:\Downloads\4th Trimester ML\Lab 10\Lab_10_MLP_XOR.html"
pdf_path = r"d:\Downloads\4th Trimester ML\Lab 10\Lab_10_MLP_XOR.pdf"

edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe"
]

browser_binary = None
for bp in edge_paths:
    if os.path.exists(bp):
        browser_binary = bp
        break

if browser_binary and os.path.exists(html_path):
    print(f"Converting HTML to PDF using {browser_binary}...")
    cmd = [
        browser_binary,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        print(f"PDF successfully generated at: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")
    else:
        print(f"PDF generation failed: {res.stderr}")
else:
    print(f"Browser binary or HTML file not found. Please ensure the notebook is exported to HTML first.")
