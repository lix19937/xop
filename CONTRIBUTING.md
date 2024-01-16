# Contributing

Below you will find a collection of guidelines for submitting issues as well as contributing code to the utils repository.
Please read those before starting an issue or a pull request.

## Issues

Specific utils design and development issues, bugs, and feature requests are maintained by GitHub Issues.

*Please do not post installation, build, usage, or modeling questions, or other requests for help to Issues.*
Use the [utils-users list]() instead.
This helps developers maintain a clear, uncluttered, and efficient view of the state of utils.
See the chapter [utils-users](#utils-users) below for guidance on posting to the users list.

When reporting an issue, it's most helpful to provide the following information, where applicable:
* How does the problem look like and what steps reproduce it?
* Can you reproduce it using the latest [master](), compiled with the `DEBUG` make option?
* What hardware and software are you running? In particular:
	* GPU make and model, if relevant,
	* operating system/distribution,
	* compiler; please also post which version (for example, with GCC run `gcc --version` to check),
	* CUDA version, if applicable (run `nvcc --version` to check),
	* cuDNN version, if applicable (version number is stored in `cudnn.h`, look for lines containing `CUDNN_MAJOR`, `CUDNN_MINOR` and `CUDNN_PATCHLEVEL`),
	* BLAS library,
* **What have you already tried** to solve the problem? How did it fail? Are there any other issues related to yours?
* If this is not a build-related issue, does your installation pass `make runtest`?
* If the bug is a crash, provide the backtrace (usually printed by utils; always obtainable with `gdb`).
* If you are reporting a build error that seems to be due to a bug in utils, please attach your build configuration (either Makefile.config or CMakeCache.txt) and the output of the make (or cmake) command.

If only a small portion of the code/log is relevant to your issue, you may paste it directly into the post, preferably using Markdown syntax for code block: triple backtick ( \`\`\` ) to open/close a block.
In other cases (multiple files, or long files), please **attach** them to the post - this greatly improves readability.

If the problem arises during a complex operation (e.g. large script using pyutils, long network prototxt), please reduce the example to the minimal size that still causes the error.
Also, minimize influence of external modules, data etc. - this way it will be easier for others to understand and reproduce your issue, and eventually help you.
Sometimes you will find the root cause yourself in the process.

Try to give your issue a title that is succinct and specific. The devs will rename issues as needed to keep track of them.

## utils-users

Before you post to the [utils-users list](), make sure you look for existing solutions.
The utils community has encountered and found solutions to countless problems - benefit from the collective experience.
Recommended places to look:
* the [users list]() itself,
* [`utils`]() tag on StackOverflow,
* [GitHub issues]() tracker (some problems have been answered there),
* the public [wiki](),
* the official [documentation]().

Found a post/issue with your exact problem, but with no answer?
Don't just leave a "me too" message - provide the details of your case.
Problems with more available information are easier to solve and attract good attention.

When posting to the list, make sure you provide as much relevant information as possible - recommendations for an issue report (see above) are a good starting point.  
*Please make it very clear which version of utils you are using, especially if it is a fork not maintained by BVLC.*

Formatting recommendations hold: paste short logs/code fragments into the post (use fixed-width text for them), **attach** long logs or multiple files.

## Pull Requests

utils welcomes all contributions.

See the [contributing guide]() for details.

Briefly: read commit by commit, a PR should tell a clean, compelling story of _one_ improvement to utils. In particular:

* A PR should do one clear thing that obviously improves utils, and nothing more. Making many smaller PRs is better than making one large PR; review effort is superlinear in the amount of code involved.
* Similarly, each commit should be a small, atomic change representing one step in development. PRs should be made of many commits where appropriate.
* Please do rewrite PR history to be clean rather than chronological. Within-PR bugfixes, style cleanups, reversions, etc. should be squashed and should not appear in merged PR history.
* Anything nonobvious from the code should be explained in comments, commit messages, or the PR description, as appropriate.

## Unknown 
