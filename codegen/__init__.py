"""Code-generation artifact handling (#23).

Conduct stores text in Job.response and single media files via Job.media_url;
a generated Rust solution is a multi-file Cargo project. This package parses a
model's output into a project, validates it, and stores it as a tarball under
the output dir so a downstream build evaluator can pull it by job id.

Python stores Rust here; it does not compile anything (that is the build
sandbox, P1.2).
"""
