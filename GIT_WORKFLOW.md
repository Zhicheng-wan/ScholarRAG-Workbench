# Git Workflow: Push Clean Version to GitHub

## Step 1: Create and Switch to New Branch

```bash
cd /Users/wangkaiwei/Documents/ucsd/cse291/ScholarRAG-Workbench

# Create new branch 'clean_version' from current state
git checkout -b clean_version
```

## Step 2: Stage All Changes (Deletions + New Files)

```bash
# Stage all deleted files (cleaned up versions)
git add -A

# This will stage:
# - All deleted directories (baseline, v19_cached, etc.)
# - New files (final_report.tex, documentation, etc.)
# - Modified PDFs
```

## Step 3: Commit Changes

```bash
git commit -m "Clean version: Keep only hybrid_optimized_v19_cached_batched

- Removed all intermediate versions (V2-V18, V20-V23)
- Removed baseline, v19, v19_cached directories
- Removed BM25 experiments and other experimental approaches
- Added final_report.tex (LaTeX project report)
- Added documentation (COMPILATION_INSTRUCTIONS.md, PROJECT_SUMMARY.md, SUBMISSION_CHECKLIST.md)
- Added data/README.md documenting final structure
- Only keeping hybrid_optimized_v19_cached_batched (production system)
- Space saved: ~15-20GB

Final system performance:
- MRR: 0.950 (+15.2% vs baseline)
- Precision@1: 0.900 (+28.6% vs baseline)
- Latency: 29ms (45% speedup)
"
```

## Step 4: Push to GitHub

```bash
# Push the new branch to GitHub
git push -u origin clean_version
```

This creates the branch on GitHub and sets up tracking.

## Step 5: Create Pull Request (PR) for Teammate Review

Option A: Using GitHub CLI (if installed):
```bash
gh pr create --base main --head clean_version --title "Clean version: Final production system only" --body "This PR cleans up the repository to keep only the final production system (hybrid_optimized_v19_cached_batched).

**Changes:**
- ✅ Removed all intermediate versions (V2-V18, V20-V23)
- ✅ Removed baseline, V19, V19_cached directories
- ✅ Removed BM25 experiments
- ✅ Added final_report.tex (LaTeX project report)
- ✅ Added comprehensive documentation
- ✅ Space saved: ~15-20GB

**Final System:**
- System: hybrid_optimized_v19_cached_batched
- MRR: 0.950 (+15.2%)
- P@1: 0.900 (+28.6%)
- Latency: 29ms (45% speedup)

**Remaining in data/:**
- hybrid_optimized_v19_cached_batched/ (production system)
- evaluation/ (test queries)
- processed/ (preprocessed docs)
- raw/ (source data)

Please review and approve for merge to main.
"
```

Option B: Using GitHub Web Interface:
1. Go to: https://github.com/Zhicheng-wan/ScholarRAG-Workbench
2. Click "Pull requests" → "New pull request"
3. Set base: `main`, compare: `clean_version`
4. Click "Create pull request"
5. Add title and description (see above)
6. Click "Create pull request"

## Step 6: After Teammate Approval - Merge to Main

### Option A: Merge via GitHub Web Interface (Recommended)
1. Go to the Pull Request on GitHub
2. Click "Merge pull request" button
3. Select merge type:
   - "Squash and merge" (clean history, single commit)
   - "Merge commit" (preserve all commits)
   - "Rebase and merge" (linear history)
4. Click "Confirm merge"
5. Delete branch `clean_version` (GitHub will prompt)

### Option B: Merge via Command Line
```bash
# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Merge clean_version into main
git merge clean_version

# Push to GitHub
git push origin main

# Optional: Delete the clean_version branch
git branch -d clean_version
git push origin --delete clean_version
```

## Step 7: Update Your Local Repository

```bash
# Switch to main
git checkout main

# Pull the merged changes
git pull origin main

# Now your main branch is up to date!
```

## Quick Reference Commands

```bash
# Check current branch
git branch

# Check status
git status

# View commit history
git log --oneline -10

# View remote branches
git branch -r

# See what will be pushed
git diff origin/clean_version
```

## Troubleshooting

### Issue: Files too large for GitHub
GitHub has a 100MB file size limit. If you get errors:

```bash
# Check large files
find . -type f -size +50M -not -path "./.git/*"

# If needed, use Git LFS for large files
git lfs install
git lfs track "*.json"
git lfs track "*.pdf"
git add .gitattributes
```

### Issue: Need to undo last commit
```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes
git reset --hard HEAD~1
```

### Issue: Branch already exists on remote
```bash
# Delete remote branch first
git push origin --delete clean_version

# Then push your new branch
git push -u origin clean_version
```

## Summary

**Complete workflow in 4 commands:**

```bash
# 1. Create branch
git checkout -b clean_version

# 2. Stage and commit
git add -A
git commit -m "Clean version: Keep only hybrid_optimized_v19_cached_batched"

# 3. Push to GitHub
git push -u origin clean_version

# 4. Create PR on GitHub (web interface or gh CLI)
# Then wait for teammate review and merge
```

---

**Current Status**: You're on branch `add_new_version` with uncommitted changes
**Next Action**: Run the commands above to create `clean_version` branch
