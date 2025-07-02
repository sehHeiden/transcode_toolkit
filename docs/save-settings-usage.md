# Using --save-settings for Transcoding

The `--save-settings` option in the video estimation command creates a detailed analysis file that can be used to optimize your transcoding workflow.

## How to Generate Settings

Run the estimation command with `--save-settings`:

```bash
transcode-toolkit video estimate /path/to/videos --save-settings
```

This creates a `.transcode_settings.json` file in the analyzed directory containing:

- **Optimal presets** for video and audio
- **Detailed per-file analysis** with savings, SSIM, and processing time estimates
- **Summary statistics** for the entire directory

## Settings File Structure

The generated `.transcode_settings.json` file contains:

```json
{
  "version": "1.0",
  "analysis_summary": {
    "total_files": 15,
    "video_files": 10,
    "audio_files": 5,
    "total_current_size_mb": 5120.5,
    "total_potential_savings_mb": 2048.2,
    "total_savings_percent": 40.0
  },
  "optimal_presets": {
    "video_preset": "h265_balanced",
    "audio_preset": "music"
  },
  "detailed_analysis": [
    {
      "file": "video1.mp4",
      "type": "video",
      "current_size_mb": 512.0,
      "best_preset": "h265_balanced",
      "savings_mb": 204.8,
      "savings_percent": 40.0,
      "predicted_ssim": 0.923,
      "processing_time_min": 5.2
    }
  ]
}
```

## Using Settings for Transcoding

### 1. Apply Recommended Presets

Use the optimal presets directly:

```bash
# Use recommended video preset
transcode-toolkit video transcode /path/to/videos --preset h265_balanced

# Use recommended audio preset  
transcode-toolkit audio transcode /path/to/audio --preset music
```

### 2. Batch Processing with Optimal Settings

Create a script to apply different presets per file type:

```bash
#!/bin/bash
SETTINGS_FILE="/path/to/videos/.transcode_settings.json"

# Extract optimal presets
VIDEO_PRESET=$(jq -r '.optimal_presets.video_preset' "$SETTINGS_FILE")
AUDIO_PRESET=$(jq -r '.optimal_presets.audio_preset' "$SETTINGS_FILE")

# Transcode with optimal settings
transcode-toolkit video transcode /path/to/videos --preset "$VIDEO_PRESET"
transcode-toolkit audio transcode /path/to/audio --preset "$AUDIO_PRESET"
```

### 3. Per-File Optimization

For advanced users, you can extract per-file recommendations:

```bash
# Get best preset for specific file
jq -r '.detailed_analysis[] | select(.file=="video1.mp4") | .best_preset' .transcode_settings.json

# List all files with their optimal presets
jq -r '.detailed_analysis[] | "\(.file): \(.best_preset)"' .transcode_settings.json
```

### 4. Quality Validation

Use the SSIM predictions to validate quality expectations:

```bash
# Find files with high quality predictions (SSIM > 0.92)
jq -r '.detailed_analysis[] | select(.predicted_ssim > 0.92) | "\(.file): SSIM \(.predicted_ssim)"' .transcode_settings.json

# Find files that might need manual review (lower SSIM)
jq -r '.detailed_analysis[] | select(.predicted_ssim < 0.90) | "\(.file): SSIM \(.predicted_ssim) - Review needed"' .transcode_settings.json
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Optimize Media Files
on:
  push:
    paths: ['media/**']

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Analyze media files
        run: |
          transcode-toolkit video estimate media/ --save-settings
          
      - name: Apply optimal transcoding
        run: |
          SETTINGS_FILE="media/.transcode_settings.json"
          VIDEO_PRESET=$(jq -r '.optimal_presets.video_preset' "$SETTINGS_FILE")
          AUDIO_PRESET=$(jq -r '.optimal_presets.audio_preset' "$SETTINGS_FILE")
          
          transcode-toolkit video transcode media/ --preset "$VIDEO_PRESET"
          transcode-toolkit audio transcode media/ --preset "$AUDIO_PRESET"
```

## Best Practices

1. **Run analysis first**: Always generate settings before transcoding large batches
2. **Review predictions**: Check SSIM scores for quality-critical content
3. **Monitor processing times**: Use time estimates for scheduling and resource planning
4. **Validate samples**: Test a few files manually before batch processing
5. **Keep settings files**: Store them for future reference and optimization tracking

## Advanced Usage

### Custom Preset Selection

You can override automatic recommendations by analyzing the detailed data:

```python
import json

# Load settings
with open('.transcode_settings.json') as f:
    settings = json.load(f)

# Find files that would benefit from higher quality preset
high_quality_files = [
    analysis for analysis in settings['detailed_analysis'] 
    if analysis['current_size_mb'] > 100 and analysis['predicted_ssim'] > 0.95
]

# Apply different preset for high-quality files
for file_info in high_quality_files:
    print(f"Consider h265_high preset for {file_info['file']}")
```

### Monitoring and Reporting

Use the settings file to generate reports:

```bash
# Generate savings report
echo "Total potential savings: $(jq -r '.analysis_summary.total_savings_percent' .transcode_settings.json)%"
echo "Storage saved: $(jq -r '.analysis_summary.total_potential_savings_mb' .transcode_settings.json) MB"

# Processing time estimate
jq -r '.detailed_analysis[] | .processing_time_min' .transcode_settings.json | awk '{sum+=$1} END {print "Estimated total processing time:", sum, "minutes"}'
```

This workflow ensures optimal transcoding results while maximizing efficiency and maintaining quality standards.
