# Project Guidelines

- [Git](#git)
    - [一些git規則](#some-git-rules)
    - [Git workflow](#git-workflow)
    - [撰寫良好的commit message](#writing-good-commit-messages)
- [Documentation](#documentation)
- [Testing](#testing)
- [Structure and Naming](#structure-and-naming)
- [Code style](#code-style)
    - [Some code style guidelines](#code-style-check)
    - [Enforcing code style standards](#enforcing-code-style-standards)

<a name="git"></a>
## 1. Git
<a name="some-git-rules"></a>

### 1.1 一些Git规则

* 要開發新功能時，新增一條 branch，也稱為 feature branch 。

    _Why：_
    > 這樣就可以保持所有與這個新功能相關的工作都在他專屬的 branch 開發完成，而不是汙染到 master branch。開發期間，可以發送多次 pull request 而不會使得還不穩定而且未完成的 code 污染 master 分支。 [read more...](https://www.atlassian.com/git/tutorials/comparing-workflows#feature-branch-workflow)

* 永遠不要將 branch（直接）push 到 `master`，請使用 Pull Request。

    _Why：_
    > 透過 Pull Request，它可以通知團隊已經完成了某個功能的開發。這樣其他成員就可以做 code review，同時互相討論所提交的功能。

* 在 Pull Request 之前，請先更新您 local 的分支並且執行 rebase。

    _Why：_
    > Rebasing will merge in the requested branch (master or develop) and apply the commits that you have made locally to the top of the history without creating a merge commit (assuming there were no conflicts). Resulting in a nice and clean history. [read more ...](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)

* 請確保在 rebasing 後解決所有 conflict，再發起 Pull Request。

* Delete local and remote feature branches after merging.

    _Why：_
    > 如果不刪除 feature branch，大量殭屍分支的存在會導致分支列表的混亂。而且還能確保只有 merge 一次到 `master`。 Feature branch 只有在這個功能還在開發中存在。

# Done

-------

# Not Done Yet

* Before making a Pull Request，make sure your feature branch builds successfully and passes all tests (including code style checks).

    _Why：_
    > 因為您即將將代碼提交到這個穩定的分支。而如果您的功能分支測試未通過，那您的目標分支的構建有很大的概率也會失敗。此外，確保在進行合併請求之前應用代碼規則檢查。因為它有助於我們代碼的可讀性，並減少格式化的代碼與實際業務代碼更改混合在一起導致的混亂問題。

* 使用 [這個](./.gitignore) `.gitignore` 文件。

    _Why：_
    > 此文件已經囊括了不應該和您開發的代碼一起推送至遠程倉庫（remote repository）的系統文件列表。另外，此文件還排除了大多數編輯器的設置文件夾和文件，以及最常見的（工程開發）依賴目錄。

* 保護您的 `develop` 和 `master` 分支。

    _Why：_
    > 這樣可以保護您的生產分支免受意外情況和不可回退的變更。更多請閱讀... [Github](https://help.github.com/articles/about-protected-branches/) 以及 [Bitbucket](https://confluence.atlassian.com/bitbucketserver/using-branch-permissions-776639807.html)

<a name="git-workflow"></a>
### 1.2 Git 工作流
基于以上原因, 我们将 [功能分支工作流](https://www.atlassian.com/git/tutorials/comparing-workflows#feature-branch-workflow) ， [交互式变基的使用方法](https://www.atlassian.com/git/tutorials/merging-vs-rebasing#the-golden-rule-of-rebasing) 结合一些 [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows#gitflow-workflow)中的基础 (比如，命名和使用一个develop branch)一起使用。 主要步骤如下:

* 针对一个新项目, 在您的项目目录初始化您的项目。 __如果是（已有项目）随后的功能开发/代码变动，这一步请忽略__。

   ```sh
   cd <项目目录>
   git init
   ```

* 检出（Checkout） 一个新的功能或故障修复（feature/bug-fix）分支。

    ```sh
    git checkout -b <分支名称>
    ```

* 新增代码变更。

    ```sh
    git add
    git commit -a
    ```

    _为什么：_
    > `git commit -a` 会独立启动一个编辑器用来编辑您的说明信息，这样的好处是可以专注于写这些注释说明。更多请阅读 *章节 1.3*。

* 保持与远程（develop分支）的同步，以便（使得本地 develop 分支）拿到最新变更。
    ```sh
    git checkout develop
    git pull
    ```

    _为什么：_
    > 当您进行（稍后）变基操作的时候，保持更新会给您一个在您的机器上解决冲突的机会。这比（不同步更新就进行下一步的变基操作并且）发起一个与远程仓库冲突的合并请求要好。

* （切换至功能分支并且）通过交互式变基从您的develop分支中获取最新的代码提交，以更新您的功能分支。

    ```sh
    git checkout <branchname>
    git rebase -i --autosquash develop
    ```

    _为什么：_
    > 您可以使用 `--autosquash` 将所有提交压缩到单个提交。没有人会愿意（看到） `develop` 分支中的单个功能开发就占据如此多的提交历史。 [更多请阅读...](https://robots.thoughtbot.com/autosquashing-git-commits)

* 如果没有冲突请跳过此步骤，如果您有冲突,  就需要[解决它们](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)并且继续变基操作。
    ```sh
    git add <file1> <file2> ...
    git rebase --continue
    ```
* 推送您的（功能）分支。变基操作会改变提交历史, 所以您必须使用 `-f` 强制推送到远程（功能）分支。 如果其他人与您在该分支上进行协同开发，请使用破坏性没那么强的 `--force-with-lease` 参数。
    ```sh
    git push -f
    ```

    _为什么:_
    > 当您进行 rebase 操作时，您会改变功能分支的提交历史。这会导致 Git 拒绝正常的 `git push` 。那么，您只能使用 `-f` 或 `--force` 参数了。[更多请阅读...](https://developer.atlassian.com/blog/2015/04/force-with-lease/)

* 提交一个合并请求（Pull Request）。
* Pull Request 会被负责代码审查的同事接受，合并和关闭。
* 如果您完成了开发，请记得删除您的本地分支。

    ```sh
    git branch -d <分支>
    ```
    （使用以下代码）删除所有已经不在远程仓库维护的分支。
    ``` sh
    git fetch -p && for branch in `git branch -vv | grep ': gone]' | awk '{print $1}'`; do git branch -D $branch; done
    ```

<a name="writing-good-commit-messages"></a>
### 1.3 如何写好 Commit Message

坚持遵循关于提交的标准指南，会让在与他人合作使用 Git 时更容易。这里有一些经验法则 ([来源](https://chris.beams.io/posts/git-commit/#seven-rules)):

 * 用新的空行将标题和主体两者隔开。

    _为什么：_
    > Git 非常聪明，它可将您提交消息的第一行识别为摘要。实际上，如果您尝试使用 `git shortlog` ，而不是 `git log` ，您会看到一个很长的提交消息列表，只会包含提交的 id 以及摘要（，而不会包含主体部分）。

 * 将标题行限制为50个字符，并将主体中一行超过72个字符的部分折行显示。

    _为什么：_
    > 提交应尽可能简洁明了，而不是写一堆冗余的描述。 [更多请阅读...](https://medium.com/@preslavrachev/what-s-with-the-50-72-rule-8a906f61f09c)

 * 标题首字母大写。
 * 不要用句号结束标题。
 * 在标题中使用 [祈使句](https://en.wikipedia.org/wiki/Imperative_mood) 。

    _为什么：_
    > 与其在写下的信息中描述提交者做了什么，不如将这些描述信息作为在这些提交被应用于该仓库后将要完成的操作的一个说明。[更多请阅读...](https://news.ycombinator.com/item?id=2079612)

 * 使用主体部分去解释 **是什么** 和 **为什么** 而不是 **怎么做**。

 <a name="文档"></a>
## 2. 文档

![文档](/images/documentation.png)

* 可以使用这个 [模板](./README.sample.md) 作为 `README.md` （的一个参考）, 随时欢迎添加里面没有的内容。
* 对于具有多个存储库的项目，请在各自的 `README.md` 文件中提供它们的链接。
* 随项目的进展，持续地更新 `README.md` 。
* 给您的代码添加详细的注释，这样就可以清楚每个主要部分的含义。
* 如果您正在使用的某些代码和方法，在github或stackoverflow上已经有公开讨论，请在您的注释中包含这些链接，
* 不要把注释作为坏代码的借口。保持您的代码干净整洁。
* 也不要把那些清晰的代码作为不写注释的借口。
* 当代码更新，也请确保注释的同步更新。

<a name="testing"></a>
## 3. 测试

![测试](/images/testing.png)

* 如果需要，请构建一个 `test` 环境.

    _为什么：_
    > 虽然有时在 `production` 模式下端到端测试可能看起来已经足够了，但有一些例外：比如您可能不想在生产环境下启用数据分析功能，只能用测试数据来填充（污染）某人的仪表板。另一个例子是，您的API可能在 `production` 中才具有速率限制，并在请求达到一定量级后会阻止您的测试请求。

* 将测试文件放在使用 `* .test.js` 或 `* .spec.js` 命名约定的测试模块，比如 `moduleName.spec.js`

    _为什么：_
    > 您肯定不想进入一个层次很深的文件夹结构来查找里面的单元测试。[更多请阅读...](https://hackernoon.com/structure-your-javascript-code-for-testability-9bc93d9c72dc)

* 将其他测试文件放入独立的测试文件夹中以避免混淆。

    _为什么：_
    > 一些测试文件与任何特定的文件实现没有特别的关系。您只需将它放在最有可能被其他开发人员找到的文件夹中：`__test__` 文件夹。这个名字：`__test__`也是现在的标准，被大多数JavaScript测试框架所接受。

* 编写可测试代码，避免副作用（side effects），提取副作用，编写纯函数。

    _为什么：_
    > 您想要将业务逻辑拆分为单独的测试单元。您必须“尽量减少不可预测性和非确定性过程对代码可靠性的影响”。 [更多请阅读...](https://medium.com/javascript-scene/tdd-the-rite-way-53c9b46f45e3)

    > 纯函数是一种总是为相同的输入返回相同输出的函数。相反地，不纯的函数是一种可能会有副作用，或者取决于来自外部的条件来决定产生对应的输出值的函数。这使得它不那么可预测。[更多请阅读...](https://hackernoon.com/structure-your-javascript-code-for-testability-9bc93d9c72dc)

* 使用静态类型检查器

    _为什么：_
    > 有时您可能需要一个静态类型检查器。它为您的代码带来一定程度的可靠性。[更多请阅读...](https://medium.freecodecamp.org/why-use-static-types-in-javascript-part-1-8382da1e0adb)


* 先在本地 `develop` 分支运行测试，待测试通过后，再进行pull请求。

    _为什么：_
    > 您不想成为一个导致生产分支构建失败的人吧。在您的`rebase`之后运行测试，然后再将您改动的功能分支推送到远程仓库。

* 记录您的测试，包括在 `README.md` 文件中的相关说明部分。

    _为什么：_
    > 这是您为其他开发者或者 DevOps 专家或者 QA 或者其他如此幸运能和您一起协作的人留下的便捷笔记。

<a name="structure-and-naming"></a>
## 4. 结构布局与命名

![结构布局与命名](/images/folder-tree.png)

* 请围绕产品功能/页面/组件，而不是围绕角色来组织文件。此外，请将测试文件放在他们对应实现的旁边。


    **不规范**

    ```
    .
    ├── controllers
    |   ├── product.js
    |   └── user.js
    ├── models
    |   ├── product.js
    |   └── user.js
    ```
    
    **规范**
    
    ```
    .
    ├── product
    |   ├── index.js
    |   ├── product.js
    |   └── product.test.js
    ├── user
    |   ├── index.js
    |   ├── user.js
    |   └── user.test.js
    ```
    
    _为什么：_
    > 比起一个冗长的列表文件，创建一个单一责权封装的小模块，并在其中包括测试文件。将会更容易浏览，更一目了然。

* 将其他测试文件放在单独的测试文件夹中以避免混淆。

    _为什么：_
    > 这样可以节约您的团队中的其他开发人员或DevOps专家的时间。

* 使用 `./config` 文件夹，不要为不同的环境制作不同的配置文件。

    _为什么：_
    > 当您为不同的目的（数据库，API等）分解不同的配置文件;将它们放在具有容易识别名称（如 `config` ）的文件夹中才是有意义的。请记住不要为不同的环境制作不同的配置文件。这样并不是具有扩展性的做法，如果这样，就会导致随着更多应用程序部署被创建出来，新的环境名称也会不断被创建，非常混乱。
    > 配置文件中使用的值应通过环境变量提供。 [更多请阅读...](https://medium.com/@fedorHK/no-config-b3f1171eecd5)

* 将脚本文件放在`./scripts`文件夹中。包括 `bash` 脚本和 `node` 脚本。

    _为什么：_
    > 很可能最终会出现很多脚本文件，比如生产构建，开发构建，数据库feeders，数据库同步等。

* 将构建输出结果放在`./build`文件夹中。将`build/`添加到`.gitignore`中以便忽略此文件夹。

    _为什么：_
    > 命名为您最喜欢的就行，`dist`看起来也蛮酷的。但请确保与您的团队保持一致性。放置在该文件夹下的东西应该是已经生成（打包、编译、转换）或者被移到这里的。您产生什么编译结果，您的队友也可以生成同样的结果，所以没有必要将这些结果提交到远程仓库中。除非您故意希望提交上去。

* 文件名和目录名请使用 `PascalCase` `camelCase` 风格。组件请使用 `PascalCase` 风格。


*  `CheckBox/index.js` 应该代表 `CheckBox` 组件，也可以写成 `CheckBox.js` ，但是**不能**写成冗长的 `CheckBox/CheckBox.js` 或  `checkbox/CheckBox.js` 。

* 理想情况下，目录名称应该和 `index.js` 的默认导出名称相匹配。

    _为什么：_
    > 这样您就可以通过简单地导入其父文件夹直接使用您预期的组件或模块。


<a name="code-style"></a>

## 5. 代码风格

![代码风格](/images/code-style.png)

<a name="code-style-check"></a>
### 5.1 若干个代码风格指导

* 对新项目请使用 Stage2 和更高版本的 JavaScript（现代化）语法。对于老项目，保持与老的语法一致，除非您打算把老的项目也更新为现代化风格。

    _为什么：_
    > 这完全取决于您的选择。我们使用转换器来使用新的语法糖。Stage2更有可能最终成为规范的一部分，而且仅仅只需经过小版本的迭代就会成为规范。

* 在构建过程中包含代码风格检查。

    _为什么：_
    > 在构建时中断下一步操作是一种强制执行代码风格检查的方法。强制您认真对待代码。请确保在客户端和服务器端代码都执行代码检查。 [更多请阅读...](https://www.robinwieruch.de/react-eslint-webpack-babel/)

* 使用 [ESLint - Pluggable JavaScript linter](http://eslint.org/) 去强制执行代码检查。

    _为什么：_
    > 我们个人很喜欢 `eslint` ，不强制您也喜欢。它拥有支持更多的规则，配置规则的能力和添加自定义规则的能力。

* 针对 JavaScript 我们使用[Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript) , [更多请阅读](https://www.gitbook.com/book/duk/airbnb-javascript-guidelines/details)。 请依据您的项目和您的团队选择使用所需的JavaScript 代码风格。

* 当使用[FlowType](https://flow.org/)的时候，我们使用 [ESLint的Flow样式检查规则。](https://github.com/gajus/eslint-plugin-flowtype)。

    _为什么：_
    > Flow 引入了很少的语法，而这些语法仍然需要遵循代码风格并进行检查。

* 使用 `.eslintignore` 将某些文件或文件夹从代码风格检查中排除。

    _为什么：_
    > 当您需要从风格检查中排除几个文件时，就再也不需要通过 `eslint-disable` 注释来污染您的代码了。

* 在Pull Request之前，请删除任何 `eslint` 的禁用注释。

    _为什么：_
    > 在处理代码块时禁用风格检查是正常现象，这样就可以关注在业务逻辑。请记住把那些 `eslint-disable` 注释删除并遵循风格规则。

* 根据任务的大小使用 `//TODO：` 注释或做一个标签（ticket）。

    _为什么：_
    > 这样您就可以提醒自己和他人有这样一个小的任务需要处理（如重构一个函数或更新一个注释）。对于较大的任务，可以使用由一个lint规则（`no-warning-comments`）强制要求其完成（并移除注释）的`//TODO（＃3456）`，其中的`#3456`号码是一个标签（ticket），方便查找且防止相似的注释堆积导致混乱。


* 随着代码的变化，始终保持注释的相关性。删除那些注释掉的代码块。

    _为什么：_
    > 代码应该尽可能的可读，您应该摆脱任何分心的事情。如果您在重构一个函数，就不要注释那些旧代码，直接把要注释的代码删除吧。

* 避免不相关的和搞笑的的注释，日志或命名。

    _为什么：_
    > 虽然您的构建过程中可能（应该）移除它们，但有可能您的源代码会被移交给另一个公司/客户，您的这些笑话应该无法逗乐您的客户。

* 请使用有意义容易搜索的命名，避免缩写名称。对于函数使用长描述性命名。功能命名应该是一个动词或动词短语，需要能清楚传达意图的命名。

    _为什么：_
    > 它使读取源代码变得更加自然。

* 依据《代码整洁之道》的step-down规则，对您的源代码文件中的函数（的声明）进行组织。高抽象级别的函数（调用了低级别函数的函数）在上，低抽象级别函数在下，（保证了阅读代码时遇到未出现的函数仍然是从上往下的顺序，而不会打断阅读顺序地往前查找并且函数的抽象层次依次递减）。

    _为什么：_
    > 它使源代码的可读性更好。

<a name="enforcing-code-style-standards"></a>
### 5.2 强制的代码风格标准

* 让您的编辑器提示您关于代码风格方面的错误。 请将 [eslint-plugin-prettier](https://github.com/prettier/eslint-plugin-prettier) 与 [eslint-config-prettier](https://github.com/prettier/eslint-config-prettier) 和您目前的ESLint配置一起搭配使用。 [更多请阅读...](https://github.com/prettier/eslint-config-prettier#installation)

* 考虑使用Git钩子。

    _为什么：_
    > Git的钩子能大幅度地提升开发者的生产力。在做出改变、提交、推送至暂存区或者生产环境的过程中（充分检验代码），再也不需要担心（推送的代码会导致）构建失败。 [更多请阅读...](http://githooks.com/)

* 将Git的precommit钩子与Prettier结合使用。

    _为什么：_
    > 虽然`prettier`自身已经非常强大，但是每次将其作为单独的一个npm任务去格式化代码，并不是那么地高效。 这正是`lint-staged`（还有`husky`）可以解决的地方。关于如何配置 `lint-staged` 请阅读[这里](https://github.com/okonet/lint-staged#configuration) 以及如何配置 `husky` 请阅读[这里](https://github.com/typicode/husky)。
