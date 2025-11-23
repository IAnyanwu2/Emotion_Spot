<#
Push workflow helper script for Windows PowerShell.
Use this script to set up Git LFS, .gitattributes, commit and push the `inference-docs` branch.
#>
param(
    [string]$remote = "https://github.com/IAnyanwu2/Emotion_Spot.git",
    [string]$branch = "inference-docs",
    [string]$message = "chore: add inference docs, parity tools, and git-lfs config"
)

# Set safe directory
git config --global --add safe.directory "$PWD"

# Ensure remote is set
try { git remote remove origin } catch { }
git remote add origin $remote

# Install/ensure Git LFS is active
if (!(Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Host "Git LFS not found. Please install it and rerun. (choco install git-lfs)"
    exit 1
}

git lfs install --skip-smudge

# Track model files
git lfs track "*.pt"
git lfs track "*.pth"

# Add changed files
git add .gitattributes
git add .gitignore
# Add all changes
git add -A

# Show status before commit
Write-Host "---- Git status: ----"
git status

# Commit
try {
    git commit -m $message
} catch {
    Write-Host "No changes to commit or commit failed. Please review git status and stage changes manually." ; exit 1
}

# Create branch if not exists
try {
    git checkout -b $branch
} catch {
    git checkout $branch
}

# Push
Write-Host "Pushing branch $branch to origin..."
git push -u origin $branch

Write-Host "Done. If you want the models as release artifacts, create a GitHub Release and upload PT/PTH files there (recommended for large files)."
