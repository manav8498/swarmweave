# Security Policy

## Reporting a Vulnerability

If you believe you've found a security vulnerability in swarmweave, please
**do not** open a public GitHub issue. Instead, email the maintainers at the
contact address listed in the repository's `README.md` with:

- A description of the vulnerability
- Steps to reproduce (a minimal proof-of-concept is ideal)
- The impact you believe it has
- Any suggested mitigations

We will acknowledge receipt within 72 hours and work with you on a coordinated
disclosure timeline.

## Supported Versions

Only the latest minor release of swarmweave receives security fixes. We
recommend always running the most recent version.

## Scope

In scope:

- The `swarmweave` Python package itself
- Official examples under `examples/`
- The CLI (`swarmweave` entry point)

Out of scope:

- Upstream dependencies (report those to their maintainers)
- User-provided tools, prompts, or data
