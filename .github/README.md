# Project Guidelines

- [Git](#git)
    - [一些git規則](#some-git-rules)
    - [Git workflow](#git-workflow)
    - [撰寫良好的commit message](#writing-good-commit-messages)
- [撰寫文件](#documentation)
- [測試](#testing)
- [Structure and Naming](#structure-and-naming)
- [Code style](#code-style)
    - [Enforcing code style standards](#enforcing-code-style-standards)

<a name="git"></a>
## 1. Git
<a name="some-git-rules"></a>

### 1.1 一些Git規則

* 要開發新功能時，就新增一條 branch，也就是 feature branch 。

    _Why：_
    > 保持此新功能相關的工作都在專屬的 branch 開發完成，可以避免不穩定或未完成的 code 汙染到 master branch。 [read more...](https://www.atlassian.com/git/tutorials/comparing-workflows#feature-branch-workflow)

* 永遠不要將 branch（直接） push 到 `master`，請使用 Pull Request。

    _Why：_
    > 透過 Pull Request，它可以通知團隊已經完成了某個功能的開發。其他成員就可以做 code review，同時互相討論所提交的功能。

* **在 Pull Request 之前，請先與 remote repository 同步您 local 的分支並且執行 rebase。**

    _Why：_
    > Rebasing will merge in the requested branch (`master` or `develop`) and apply the commits that you have made locally to the top of the history without creating a merge commit (assuming there were no conflicts). Resulting in a nice and clean history. [read more ...](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)

* 請確保在 rebase 後解決所有 conflict，再發起 Pull Request。

* Delete local and remote feature branches after merging.

    _Why：_
    > 如果完成此功能後不刪除 feature branch，大量殭屍分支會導致 branch list 的混亂。

    > 保持刪除習慣能確保 feature branch 只會 merge 一次到 `master` or `develop`。

    > Feature branch 只應該在此功能還在開發時存在。

* Before making a Pull Request, make sure your feature branch builds successfully and passes all tests (including code style checks).

    _Why：_
    > 因為您即將將 code push 到穩定的 branch。而如果您的 feature branch 未通過 test，那您的目標分支也很有可能會失敗。
    
    > 也確保 pull request 之前檢查過 code style，增加程式碼的可讀性。

* 使用 [這個](./.gitignore) `.gitignore` 文件。

    _Why：_
    > 此文件會排除所有不應該 push 至 remote repository 的資料夾和檔案。例如：編輯器設定檔、pycache、容量極大的交易數據csv、API私鑰...。

* 保護 `master` 分支不會受到意外情況和不可回復的變更。 read more... [Github](https://help.github.com/articles/about-protected-branches/) 以及 [Bitbucket](https://confluence.atlassian.com/bitbucketserver/using-branch-permissions-776639807.html)

<a name="git-workflow"></a>
### 1.2 Git workflow
基於以上原因, 我們將 [Feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows#feature-branch-workflow) ， [Rebase 的使用方法](https://www.atlassian.com/git/tutorials/merging-vs-rebasing#the-golden-rule-of-rebasing) 結合一些 [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows#gitflow-workflow) 中的基礎。

主要步驟如下:

* 切換分支，使用 checkout 至一個新的功能或故障修復（feature/bug-fix）分支：

    ```sh
    git checkout -b <分支名稱>
    ```

* 新增變更：

    ```sh
    git add <file1> <file2> ...
    git commit
    ```

    _Why：_

    > git add <file1> <file2> ...  - you should add only files that make up a small and coherent change.

    > git add 會把檔案交給 Git ，讓 Git 開始「追蹤」目錄，此時內容會被加到暫存區。

    > git commit 會開啟一個 editor ，可以將此次更新的相關訊息寫在裡面。
    
    > git commit 同時會將暫存區的內容提交到儲存庫（Repository）保存。

    > Read more about it in section 1.3.

* 平時就保持與remote branch（master or develop）的同步：
    ```sh
    git checkout develop
    git pull
    ```

    _Why：_
    > 如此一來，當您之後進行 rebase 的時候，就可以產生較少的 conflict。

* （切換至功能分支並且）透過 rebase 從您的 develop 分支中獲取最新的代碼提交，以更新您的功能分支：

    ```sh
    git checkout <branchname>
    git rebase -i --autosquash develop
    ```

    _Why：_
    > 您可以使用 `--autosquash` 將所有提交壓縮到單個提交。沒有人會願意（看到） `develop` 分支中的單個功能開發就佔據如此多的提交歷史。 [更多請閱讀...](https://robots.thoughtbot.com/autosquashing-git-commits)

* 如果沒有衝突請跳過此步驟，如果您有衝突,  就需要[解決它們](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)並且繼續變基操作。
    ```sh
    git add <file1> <file2> ...
    git rebase --continue
    ```
* 推送您的（功能）分支。變基操作會改變提交歷史, 所以您必須使用 `-f` 強制推送到遠程（功能）分支。如果其他人與您在該分支上進行協同開發，請使用破壞性沒那麼強的 `--force-with-lease` 參數。
    ```sh
    git push -f
    ```

    _Why：_
    > 當您進行 rebase 操作時，您會改變功能分支的提交歷史。這會導致 Git 拒絕正常的 `git push` 。那麼，您只能使用 `-f` 或 `--force` 參數了。 [更多請閱讀...](https://developer.atlassian.com/blog/2015/04/force-with-lease/)

* 提交一個合併請求（Pull Request）。
* Pull Request 會被負責代碼審查的同事接受，合併和關閉。
* 如果您完成了開發，請記得刪除您的本地分支。

    ```sh
    git branch -d <分支>
    ```
    （使用以下代碼）刪除所有已經不在遠程倉庫維護的分支。
    ``` sh
    git fetch -p && for branch in `git branch -vv | grep ': gone]' | awk '{print $1}'`; do git branch -D $branch; done
    ```

# Done

-------

# Not Done Yet

<a name="writing-good-commit-messages"></a>
### 1.3 如何寫好 Commit Message

堅持遵守關於 commit 的標準指南，會讓在與他人合作使用 Git 時更容易。這裡有一些[經驗法則](https://chris.beams.io/posts/git-commit/#seven-rules):

 * commit 應該盡可能簡潔明瞭，而不是寫一堆冗餘的描述。 [更多請閱讀...](https://medium.com/@preslavrachev/what-s-with-the-50-72-rule-8a906f61f09c)
 * 標題首字母大寫。
 * 不要用句號結束標題。
 * 在標題中使用 [祈使句](https://en.wikipedia.org/wiki/Imperative_mood) 。

    _Why：_
    > 與其描述提交者做了什麼，不如描述這些 commit 完成後的操作的一個說明。 [更多請閱讀...](https://news.ycombinator.com/item?id=2079612)

 * 使用主體部分去解釋 **是什麼** 和 **為什麼** 而不是 **怎麼做**。

<a name="documentation"></a>
## 2. 撰寫文件

* 持續更新 `README.md` 。
* 在你開發的程式碼中添加詳細的 comment ，確保每個重要部分的意義。
* 如果你使用的程式碼或方法在github或stackoverflow上已經有公開討論，請在註釋中添加這些連結，
* 當程式碼更新時，確保註釋也有同步更新。

<a name="testing"></a>
## 3. 測試

* 如果需要，請特別建構一個 `test` 環境.

    _Why：_
    > 雖然有時在 `production` 模式下端到端測試可能看起來已經足夠了，但有一些例外：比如您可能不想在生產環境下啟用數據分析功能，只能用測試數據來填充（污染）某人的儀表板。另一個例子是，您的API可能在 `production` 中才具有速率限制，並在請求達到一定量級後會阻止您的測試請求。

* 將測試文件放在使用 `* .test.py` 命名的測試模塊，比如 `moduleName.test.py`

    _Why：_
    > 您肯定不想進入一個很深層的文件夾來尋找裡面的單元測試。 [read more...](https://hackernoon.com/structure-your-javascript-code-for-testability-9bc93d9c72dc)

* Write testable code, avoid side effects, extract side effects, write pure functions

    _Why：_
    > You want to test a business logic as separate units. You have to "minimize the impact of randomness and nondeterministic processes on the reliability of your code". [更多請閱讀...](https://medium.com/javascript-scene/tdd-the-rite-way-53c9b46f45e3)

    > A pure function is a function that always returns the same output for the same input. Conversely, an impure function is one that may have side effects or depends on conditions from the outside to produce a value. That makes it less predictable [更多請閱讀...](https://hackernoon.com/structure-your-javascript-code-for-testability-9bc93d9c72dc)

* 使用static type checker

    _Why：_
    > 有時您可能需要一個static type checker。它為您的代碼帶來一定程度的可靠性。 [更多請閱讀...](https://medium.freecodecamp.org/why-use-static-types-in-javascript-part-1-8382da1e0adb)


* 先在本地分支運行測試，測試通過後，再進行pull request。

* 記錄您的測試，包括在 `README.md` 文件中的相關說明部分。

<a name="structure-and-naming"></a>
## 4. Structure and Naming

https://github.com/yngvem/python-project-structure/blob/master/README.rst

* 使用 `./config` 文件夾，不要為不同的環境製作不同的配置文件。

    _Why：_
    > 當您為不同的目的（數據庫，API等）分解不同的配置文件;將它們放在具有容易識別名稱（如 `config` ）的文件夾中才是有意義的。請記住不要為不同的環境製作不同的配置文件。這樣並不是具有擴展性的做法，如果這樣，就會導致隨著更多應用程序部署被創建出來，新的環境名稱也會不斷被創建，非常混亂。
    > 配置文件中使用的值應通過環境變量提供。 [更多請閱讀...](https://medium.com/@fedorHK/no-config-b3f1171eecd5)


* 將構建輸出結果放在`./build`文件夾中。將`build/`添加到`.gitignore`中以便忽略此文件夾。

    _Why：_
    > 命名為您最喜歡的就行，`dist`看起來也蠻酷的。但請確保與您的團隊保持一致性。放置在該文件夾下的東西應該是已經生成（打包、編譯、轉換）或者被移到這裡的。您產生什麼編譯結果，您的隊友也可以生成同樣的結果，所以沒有必要將這些結果提交到遠程倉庫中。除非您故意希望提交上去。

* 文件名和目錄名請使用 `PascalCase` `camelCase` 風格。組件請使用 `PascalCase` 風格。


*  `CheckBox/index.js` 應該代表 `CheckBox` 組件，也可以寫成 `CheckBox.js` ，但是**不能**寫成冗長的 `CheckBox/CheckBox.js` 或  `checkbox/CheckBox.js` 。

* 理想情況下，目錄名稱應該和 `index.js` 的默認導出名稱相匹配。

    _Why：_
    > 這樣您就可以通過簡單地導入其父文件夾直接使用您預期的組件或模塊。


<a name="code-style"></a>

## 5. code style

* 在構建過程中包含代碼風格檢查。

    _Why：_
    > 在構建時中斷下一步操作是一種強制執行代碼風格檢查的方法。強制您認真對待代碼。請確保在客戶端和服務器端代碼都執行代碼檢查。 [更多請閱讀...](https://www.robinwieruch.de/react-eslint-webpack-babel/)

* 使用 [ESLint - Pluggable JavaScript linter](http://eslint.org/) 去強制執行代碼檢查。

    _Why：_
    > 我們個人很喜歡 `eslint` ，不強制您也喜歡。它擁有支持更多的規則，配置規則的能力和添加自定義規則的能力。

* 針對 JavaScript 我們使用[Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript) , [更多請閱讀](https://www.gitbook.com/book/duk/airbnb-javascript-guidelines/details)。請依據您的項目和您的團隊選擇使用所需的JavaScript 代碼風格。

* 當使用[FlowType](https://flow.org/)的時候，我們使用 [ESLint的Flow樣式檢查規則。 ](https://github.com/gajus/eslint-plugin-flowtype)。

    _Why：_
    > Flow 引入了很少的語法，而這些語法仍然需要遵循代碼風格並進行檢查。

* 使用 `.eslintignore` 將某些文件或文件夾從代碼風格檢查中排除。

    _Why：_
    > 當您需要從風格檢查中排除幾個文件時，就再也不需要通過 `eslint-disable` 註釋來污染您的代碼了。

* 在Pull Request之前，請刪除任何 `eslint` 的禁用註釋。

    _Why：_
    > 在處理代碼塊時禁用風格檢查是正常現象，這樣就可以關注在業務邏輯。請記住把那些 `eslint-disable` 註釋刪除並遵循風格規則。

* 根據任務的大小使用 `//TODO：` 註釋或做一個標籤（ticket）。

    _Why：_
    > 這樣您就可以提醒自己和他人有這樣一個小的任務需要處理（如重構一個函數或更新一個註釋）。對於較大的任務，可以使用由一個lint規則（`no-warning-comments`）強制要求其完成（並移除註釋）的`//TODO（＃3456）`，其中的`#3456`號碼是一個標籤（ticket），方便查找且防止相似的註釋堆積導致混亂。


* 請使用有意義容易搜索的命名，避免縮寫名稱。對於函數使用長描述性命名。功能命名應該是一個動詞或動詞短語，需要能清楚傳達意圖的命名。

    _Why：_
    > 它使讀取源代碼變得更加自然。

* 依據《代碼整潔之道》的step-down規則，對您的源代碼文件中的函數（的聲明）進行組織。高抽象級別的函數（調用了低級別函數的函數）在上，低抽象級別函數在下，（保證了閱讀代碼時遇到未出現的函數仍然是從上往下的順序，而不會打斷閱讀順序地往前查找並且函數的抽象層次依次遞減）。

    _Why：_
    > 它使源代碼的可讀性更好。

<a name="enforcing-code-style-standards"></a>
### 5.1 強制 code style 標準

* 讓您的編輯器提示您關於代碼風格方面的錯誤。請將 [eslint-plugin-prettier](https://github.com/prettier/eslint-plugin-prettier) 與 [eslint-config-prettier](https://github.com/prettier/eslint-config-prettier) 和您目前的ESLint配置一起搭配使用。 [更多請閱讀...](https://github.com/prettier/eslint-config-prettier#installation)

* 使用Git hooks：

    _Why：_
    > Git hooks 能大幅度地提升 dev 的生產力。在做出 change, commit, or push to staging，再也不用擔心（push 的 code 會導致）build 失敗。 [read more...](http://githooks.com/)
