<img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# Contributing to [Industry Solution Accelerators](https://www.databricks.com/solutions/accelerators) - the Field Guide

Thank you for your interest in contributing to solution accelerators! Solution accelerator are Databricks' repository to host reusable technical assets for industry technical challenges and business use cases. The program is run by Sales GTM Verticals and supported by field contribution.

The purpose of this notebook is to describe the process for contributing to accelerators and provide helpful checklists for important milestones: intake, commit, standardization and publication. Hopefully this checklist will be useful for first-time and repeat contributors alike.

<img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/Solution%20Accelerator%20FY24.jpg'></img>

___
Maintainer: [@nicole.lu](https://databricks.enterprise.slack.com/team/jingting_lu)
___


## Intake
❓ If you brought your own code, can you summarize what problem your code solves in less than 100 words? 
  * Does it tackle an industry **business use case**, or a common industry **technical challenge**

❓  Have you discussed the topic with a Technical Director? If you are not sure which vertical your work is best suited for, contact [@nicole.lu](https://databricks.enterprise.slack.com/team/jingting_lu) for an intake consultation. 

**The Technical Directors will approve the accelerator and place it on a publication roadmap for their industry.** The Technical Directors are:
  * Retail CPG: Bryan Smith 
  * Financial Services: Antoine Amend, Eon Retief
  * Media Entertainment: Dan Morris
  * Health Life Sciense: Amir Kermany, Aaron Zarova
  * Manufacturing: Bala Amavasai
  * Cyber Security: Lipyeow Lim
  * Public Sector: No Technical Director but Field Eng owns content curation and development. Reach out to Milos Colic
  
❓ Do we have the rights to use the source datasets and libraries in your code?
- Please fill out the dependency-license table in the README. Make sure our dependencies are open source. If we need to use some written documentation to substantiate our rights to use any data or code dependency, file a legal review ticket 
- If you need to synthesize and store some source data, use a publically accessible cloud storage, such as `s3://db-gtm-industry-solutions/data/`

❓ Is the code reusable by a broad array of customers? No customer-specific implementation details please.

❓  Do you know the scope of work for this accelerator? 
  * At the minimum, you are responsible for making the code in the repo to tell a cohesive story
  * You may need to provide a blog post, video recording or slides. The technical director will discuss and decide which **publishing tier** the accelerator will be launched at. The **publishing tier** determines the full scope and the list of final deliverables for the accelerator. Higher tiers may require a blog post, slides, video recording and more. The industry vertical will lean in with marketing resources if they decide to publish the accelerator at a higher tier 💪
  
<img src='https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/Solution%20Accelerator%20FY24%20(1).jpg'></img>
___


## Commit: Before the First Code Review

❓  Do you know how you will be collaborating with reviewers and other contributors on this code? 
  *  You may collaborate with the reviewer/contributor in the same workspace
  *  You may also receive a repo for the accelerator in https://github.com/databricks-industry-solutions to collaborate on via pull requests

❓ Do we have rights to use the source data and dependencies? Do we need to host data?
- Please fill out the dependency-license table in the README. Make sure our dependencies are open source. If we need to use some written documentation to substantiate our rights to use any data or code dependency, file an LPP (legal review) ticket 
- If you need to synthesize and store some source data, use a publically accessible cloud storage, such as `s3://db-gtm-industry-solutions/data/`

❓ Does the code contain any credentials? If yes, **scrub** the credentials from your code. Contact [@nicole.lu](https://databricks.enterprise.slack.com/team/jingting_lu) to set up secrets in demo and testing workspaces. Prepare a short paragraph describing how the user would set up the dependencies and collect their own credentials 

❓ Have you explored https://github.com/databricks-industry-solutions/industry-solutions-blueprints? This repo illustrates a compulsory directory standard. All new accelerator repos are created with this template in place. If you are provided a repo for collaboration, please commit your code according to this template.

- **Narrative notebooks** are stored on the top level and **numbered**. 
- **The RUNME notebook** is the entry point of your accelerator. It creates the job and clusters your user will use to run the notebooks, acting as the definition of the integration test for this accelerator. All published solution accelerator run nightly integration tests
- **Util and configuration notebooks** can be stored `./util` and `./config` directories.  Example util notebooks for common tasks such as **preparing source data** and **centralizing configuration** are available in this repo and they are reused in almost every accelerator. You can save time by modifying and reusing these standard components.
- **Dashboards** can be saved in `./dashboard` directory and created in the `RUNME` notebook. See an example in the `RUNME` notebook in this repository. The dashboard import feature is in private preview and enabled on the [e2-demo-field-eng workspace](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485).
- **Images and other arbitrary files** can be stored in `./images/` and `./resources/` directories if they are not large (less than 1 mb). Imagines can be embedded via its Github url, which will work once the repository is made public. Do not use relative paths like `./images/image.png` in either notebooks or the README.md file - see the images throughout this notebook for examples.  Larger resources can be stored in a public storage account, such as , such as `s3://db-gtm-industry-solutions/`

___

## Standardization: Before Reviewing with Technical Directors and other Collaborators

❓ Have you read a few accelerators and familiarized with the style? Here are some great recent examples: [IOC matching accelerator](https://github.com/databricks-industry-solutions/ioc-matching) from Cyber Security, [Pixels accelerator](https://github.com/databricks-industry-solutions/pixels) from HLS, [ALS Recommender accelerator](https://github.com/databricks-industry-solutions/als-recommender) from RCG.

❓ Have you tested the code end-to-end? 
  * Set up the multi-task job in `RUNME` by modifying the sample job json - the job defines the workflow you intend the user to run in their own workspace
  * Run the RUNME notebook to generate your accelerator workflow. Run the workflow end-to-end to show that all code runs for the accelerator.
  * Create a [**pull request**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) from your branch into main. The creation of pull request triggers integration tests in multiple workspaces. [Example](https://github.com/databricks-industry-solutions/media-mix-modeling/actions)
  
❓ Have you resolved all integration test errors? If you have issues seeing integration run histories for debugging, slack [@nicole.lu](https://databricks.enterprise.slack.com/team/jingting_lu) for help.

___

## Publication: Before the Content is Made Publically Visible

Accelerators must be reviewed with the sponsoring Technical Director and 1 other technical SME. The SME can be a lead of the SME groups (ML-SME etc) or a vertical lead SA. 

❓ Have you resolved all integration test errors? 

❓ Does your accelerator have in-depth discussion with at least one focus: **business use case**, **industry technical challenge** or both

❓ Does the notebook(s) explain the business use case and the technical pattern via sufficient Markdowns?

❓ Did you work with the Industry Marketers to publish other marketing assets such as blogs? 
* RCG, MFG: Sam Steiny
* FSI, Cyber Security: Anna Cuisia
* CME: Bryan Saftler
* HLS: Adam Crown
* Public Sector: Lisa Sion

---

 If your answers are yes to all the above ... 
## 🍻 Congratulations! You have successfully published a solution accelerator. 

Your thought leadership  
* Is visible on the Databricks [website](https://www.databricks.com/solutions/accelerators)
* May be showcased on our Marketplace
* May be used in training material
* Maybe implemented by our Professional Services, Cloud Partners, SIs and have many more channels of influence.

___

## Maintenance, Feedback and Continued Improvement
❗ If you know of a customer who benefited from an accelerator, you or the account team should fill out the customer use capture form [here](https://docs.google.com/forms/d/1Seo5dBNYsLEK7QgZ1tzPvuA9rxXxr1Sh_2cwu9hM9gM/edit) 📋

❗ You can track which Customer Accounts imported your accelerator if you have [logfood](https://adb-2548836972759138.18.azuredatabricks.net/sql/dashboards/b85f5b93-2e4c-40ee-92fd-9b30d1d8a659?o=2548836972759138#) access. 📈

❗ [@nicole.lu](https://databricks.enterprise.slack.com/team/jingting_lu) may reach out for help if some hard-to-resolve bugs arose from nightly testing 🪲

❗ Users may open issues to ask questions about the accelerator. Users may also contribute to solution accelerators as long as they accept our Contributing License Agreement. We have an automated process in place and the external collaborator can accept the Contributing License Agreement on their own. 🤝
