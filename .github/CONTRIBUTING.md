# How to Contribute

## Providing Feedback

Issue reports and feature proposals are very welcome.
Please use the [GitHub issue tracker](https://github.com/cmelab/grits/issues/) for this.

## Contributing Code

Contributions are welcomed via pull requests (PRs) on GitHub. Developers and/or users will review requested changes and make comments. The rest of this file will serve as a set of general guidelines for contributors.

### Guideline for Code Contributions

* Use the [OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow) model of development:
  - Both new features and bug fixes should be developed in branches based on `main`.
* Avoid introducing dependencies -- especially those that might be harder to install in high-performance computing environments.
* Create [unit tests](https://en.wikipedia.org/wiki/Unit_testing) for any added features that cover the common and corner cases of the code.
* Preserve backwards-compatibility whenever possible, and make clear if something must change.
* Document any portions of the code that might be less clear to others, especially to new developers.
* Use inclusive language in all documentation and code. The [Google developer documentation style guide](https://developers.google.com/style/inclusive-documentation) is a helpful reference.

### Code Style

The [pre-commit tool](https://pre-commit.com/) is used to enforce code style guidelines. Use `pip install pre-commit` to install the tool and `pre-commit install` to configure pre-commit hooks.
