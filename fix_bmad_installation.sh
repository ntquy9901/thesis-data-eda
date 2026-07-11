# BMAD Installation Fix for Vietnam Stock Analysis Project

## Problem
BMAD installation fails in non-interactive mode because `--tools` parameter is required when no tools are previously configured.

## Solution

### Step 1: Navigate to project directory
```bash
cd D:\bmad-projects\thesis\data_eda
```

### Step 2: Run installation with correct parameters

#### Option A: Non-interactive (Recommended for automation)
```bash
npx bmad-method install --modules bmm --tools claude-code --yes
```

#### Option B: Interactive (If you want to choose options)
```bash
npx bmad-method install
# Then select:
# - Install directory: D:\bmad-projects\thesis\data_eda (default)
# - Tools: claude-code
# - Language: Vietnamese (recommended)
```

### Step 3: Verify installation
```bash
npx bmad-method status
```

Expected output should show:
- Installation directory: D:\bmad-projects\thesis\data_eda
- Modules: bmm (Build Module)
- Tools: claude-code
- Status: Installed

## Alternative: Manual Installation Commands

If npx has issues, try:
```bash
# Using npm directly
npm install -g bmad-method
bmad-method install --modules bmm --tools claude-code --yes
```

## Troubleshooting

### Issue: "No BMAD installation manifest found"
**Solution:** Run the install command again with correct parameters

### Issue: Permission errors
**Solution:** Run as administrator or use elevated privileges

### Issue: Network timeout
**Solution:** Check internet connection, try again with `--timeout` parameter

## Post-Installation Steps

After successful installation:
1. Verify BMAD directory structure
2. Check configuration files
3. Test with simple command
4. Start using bmad-help functionality

## Next Steps After Installation

1. Implement Vietnamese NLP pipeline
2. Setup midnight batch processing
3. Start correlation analysis

---
**Last Updated:** 2026-07-11
**Project:** Vietnam Stock Market Analysis
