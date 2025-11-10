#### Kotaemon Project : a DATA4GOOD git subtree folder


The folder 'kotaemon' is a git subtree folder, that can be synchronized with the 'root' common project : https://github.com/dataforgoodfr/kotaemon#

This 'root' project hosts all the common / generic tools built by Data4Good.

When you make a code change in your 'local' project, all the code alterations that are included in the 'root' project (the 'kotaemon' folder) could be push to contribute to the 'root' subtree project. (by the use of your intermediary branch like "13_ecoskills_version")

Inversely, you can regularly pull the 'root' main/branch changes.


#### Synchronisation commands

To pull the recent changes of your branch / or MAIN branch, from the 'root' common project : https://github.com/dataforgoodfr/kotaemon#

(of course, these are the MAIN branch changes that you commonly want to pull... The changes of your branch should be already shared within you team repo github of the project)

```git subtree pull --prefix=rag_system/kotaemon https://github.com/dataforgoodfr/kotaemon.git [MY_BRANCH or MAIN] --squash```

Contributing to the "root" project :

To contribute within your branch :

```git subtree push --prefix=rag_system/kotaemon  https://github.com/dataforgoodfr/kotaemon.git [MY_BRANCH]```

Replace [MY_BRANCH] with your branch version : 13_ecoskills_version, 13_sufficiency_version ...

And if the changes are very generics => Do a Merge Request to the MAIN branch with GitHub.

Be Careful ! All changes must be 'generics' to satisfy all projects ... (or explicitly written as '_example' )

... Please, Exclude "taxonomy" libs from your MR !...


###### Little Tips

Loccaly, you can create a git alias for the commands above ... Example (with the main branch of the subtree Project) for the pull command :

```git config --global alias.st-[MY_BRANCH or MAIN]-pull 'subtree pull --prefix=rag_system/kotaemon https://github.com/dataforgoodfr/kotaemon.git [MY_BRANCH or MAIN] --squash' ```

and for the push command : 

```git config --global alias.st-[MY_BRANCH or MAIN]-push 'subtree push --prefix=rag_system/kotaemon  https://github.com/dataforgoodfr/kotaemon.git [MY_BRANCH] ' ```

Now, you can use these alias :


```git st-[MY_BRANCH]-pull ```

or 

```git st-[MY_BRANCH]-push ```
