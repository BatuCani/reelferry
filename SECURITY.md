# Security policy

Security fixes target the current `main` branch and, once releases exist, the latest release. Older builds may need to update before receiving a fix.

Please report a suspected vulnerability through [GitHub private vulnerability reporting](https://github.com/BatuCani/reelferry/security/advisories/new). Do not include exploit details, provider keys, private videos, or diagnostic logs in a public issue. Include the affected version, steps to reproduce with dummy data, and expected impact. Maintainers will coordinate a private response and disclosure with you.

Reelferry stores provider keys using Electron secure storage, runs the clipping pipeline locally, and calls OpenRouter and, when connected, Zernio with the user's own accounts. Review [the data flow](docs/ARCHITECTURE.md) when assessing a report. If private reporting is unavailable, open a public issue requesting a private contact channel without vulnerability details.
