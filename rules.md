# Rules

The competition consists of two stages: the Preliminary Round and the Finals. The list of finalists advancing to the Finals will be announced after the Preliminary Round concludes.

All team members are required to attend the designated technical workshops for each competition stage in full, including both the Preliminary Round Technical Workshop and the Final Round Technical Workshop (held either on-site or online, as announced by the organizer). Full attendance by all members is mandatory for each workshop. Teams that fail to ensure the full participation of their representatives will be disqualified from the competition.

The Preliminary Round will take place from August 11 to September 1, and the Final Round will take place from September 3 to September 17. During this period, the maximum number of daily uploads is 3, and the final upload of the day will serve as your score for that day. In the event of identical scores, rankings will be determined based on the submission time; earlier submissions will rank higher.

Participating teams are strictly prohibited from using multiple accounts, manually labeling test data, attacking the leaderboard system, or using any other unfair methods to manipulate the competition results. Violators will be disqualified from the competition and prize eligibility.

Participants are allowed to use open-source data augmentation datasets and open-source pre-trained models, provided that all resources used are publicly available under open-source licenses. The sources of these data augmentation datasets and pre-trained models must be explained in a document and submitted along with the results. Participants may also utilize the baseline model provided by our institute as their pre-trained model.

The data, technology, and source code used in the competition entries must be original to the participating team or legally authorized. If any infringement of intellectual property rights is verified by the organizer, the team will be disqualified from the competition and prizes, and the team shall bear all relevant legal responsibilities.

Winning teams must submit model weights, source code, execution instructions, and related documentation as requested by the organizer for verification and acceptance.

Participating teams must achieve an mAP@0.5 of 0.6 or higher in the Preliminary Round to qualify for the Finals. If fewer than 10 teams meet this standard, the Finals may be left unawarded or the format may be adjusted at the organizer's discretion. The Final Round will take place from September 3 to September 17.

Training rules: Models must be trained using the original 34 source categories.

To ensure fairness, regular employees, contract personnel, and individuals assisting in the execution of this competition from the organizing and co-organizing institutions (including the National Academy of Marine Research, DIGITIMES, ITRI, and AWS) are not eligible to register.

The organizer reserves the right to adjust the competition system, data release methods, evaluation rules, and announcement details.

Matters not covered herein will be resolved through mutual consultation based on the situation at the time.

By completing the registration, all participants are deemed to have read and agreed to abide by the competition regulations and all relevant rules.

In case of any dispute, the organizer reserves the right of final interpretation of the event.

Please read the Supplementary Provisions for other competition regulations.

## Preliminary Round & Finals

The evaluation metric uses mean Average Precision (mAP)[1] at an Intersection over Union (IoU)[2] threshold of 0.5. A prediction bounding box is considered a True Positive (TP) if its IoU with the ground truth bounding box is greater than 0.5; otherwise, it is a False Positive (FP). Precision is then calculated based on TP and FP counts. The system evaluates the AP score for each object type and then averages the AP values across the 33+1 classes of marine debris objects to obtain the final mAP evaluation value, which determines the ranking. The system uses the COCO API[3] to calculate the mAP values.

## Best Lightweight Optimization Award

To encourage teams to develop models that balance practical application, high performance, and low resource consumption, the "Best Lightweight Optimization Award" has been specially established. This award utilizes the NetScore metric for comprehensive evaluation. All teams that successfully advance to the Finals and submit a valid model complying with the regulations will automatically qualify for this award.

The evaluation formula is defined as：Ω = 20 log₁₀(a² / (√p √m))

- 𝑎（Model Accuracy）: Measured as a percentage of mAP. For example, if the mAP is 0.85, then $a = 85$
- 𝑝（Model Parameter Count / Size）: Evaluated based on the model size in MB
- 𝑚（Computational Complexity）: Evaluated based on GFLOPs

## Supplementary Provisions

The "2026 International Marine Debris Image AI Recognition Challenge" (the "Event") is organized by the National Academy of Marine Research (the "Organizer"). In the event of any violation of these terms, the Organizer reserves the right to disqualify the team and revoke its prize eligibility. Those unwilling to accept these terms may freely withdraw from the selection process before the evaluation begins, provided that they notify the Organizer in writing.

Participating teams guarantee that all information provided, filled in, and competition content submitted are true and free from any infringement of others' rights (including but not limited to portrait rights, copyrights, and/or other intellectual property rights), and original, not involving plagiarism or ghostwriting. If any falsehood or infringement is discovered or reported and confirmed, the Organizer may disqualify the entry during the event. For winning entries, the award status will be revoked, and the winners must return the prize money and certificates. If a third party claims rights against the Organizer, the participating team shall be responsible for resolving the dispute and bearing all losses and expenses incurred by the Organizer (including but not limited to attorney fees, litigation costs, and settlement fees), which have no relation to the Organizer. In case of disputes over rights and the chronology of creation, the participating team bears the burden of proof and shall raise no objection to the Organizer's final decision.

Competition entries must not be repetitive works that have previously won awards in domestic or international competitions, and the same work must not participate in other domestic or international competitions during this competition period.

The intellectual property rights of the entries in this competition belong to the participating teams. Participants enjoy the right of first refusal to negotiate cooperation with the proposition enterprise regarding the proposal itself and derived intellectual property rights transfer or licensing matters.

If the maximum number of participating teams is exceeded, admission will be determined based on the registration timestamp.

This competition designates AWS as the development environment, providing an exclusive workspace for the marine debris competition, which will be open during the competition period. After the competition concludes, the organizer/host will terminate the AWS cloud service environment provided for the competition. The organizer/host is not responsible for backing up or preserving your data on the platform; please back up your files on your own before the event ends.

If a participating team's entry involves violence, pornography, defamation, insults, or other content detrimental to social morality or social justice, the organizer/host reserves the right to terminate the entry's participation.

All teams entering the finals and the allocation of awards will be decided by the judging panel based on the quality of the entries. If necessary, slots may be "added" or "left unawarded", and the panel may also decide to alter award names.

Participating teams agree to license materials provided for the competition (including but not limited to solution presentations, videos, and other outputs) to the Organizer free of charge for various promotional purposes related to this event. If a third party's intellectual property or other rights (including but not limited to portrait rights) are used in the competition content, the participating team must confirm and warrant that there is no unlawful infringement, and that they have obtained the third party's permission to sub-license the materials to the Organizer for an indefinite period within the Taiwan region, utilizing methods including but not limited to public transmission, distribution, public display, and public publication.

Participating teams and their members should act in a manner that maintains the Organizer's reputation. If any issues arise, please contact the project staff proactively to seek a resolution. No defamation should occur before facts are clarified.

The organizer/host will fulfill its duty of confidentiality regarding the personal data filled out by participants and will never leak it. Please fill it out with peace of mind.

Registering for this competition through this event website indicates that you have agreed to the Personal Data Authorization Terms of this event.

Winners must provide complete award-receiving documents required by the Organizer and complete the award process within the designated time. Those who fail to complete the procedures within the specified time will be deemed to have waived their winning qualifications.

According to the tax laws of the Republic of China, a participating team is subject to a mandatory withholding tax of 10% to 20% on the prize received (the prize money provided by the Organizer is tax-inclusive); if the winner fails to pay the tax in accordance with regulations, the Organizer may cancel their prize eligibility. The Organizer will not interfere with the internal distribution of prizes within the team.

AI Model and Source Code Licensing: The top three winning teams must grant a non-exclusive license for the intellectual property rights of their AI model files and source code to the Academy for academic, public welfare, and official purposes. Under no circumstances will it be used for commercial purposes.

Personal Data Use Policy: Participants agree that DIGITIMES and partner units (including organizers, co-organizers, technical support partners, and partners) will jointly retain your personal information. In accordance with the Personal Data Protection Act, for the purposes of executing, advertising, and marketing of the "2026 International AI Challenge for Marine Debris Image Recognition," personal data provided by participants will be collected, processed, and utilized within the Taiwan region until the participant proactively requests the organizer to delete or stop processing and utilizing their personal data. Participants have the right to request the organizer to access, provide copies of, or correct their personal data at any time, and may also contact the organizer at any time to object to the continued collection, processing, or utilization of their personal data. If a participant disagrees to provide or provides incorrect personal data, the organizer will cancel the participant's eligibility to participate or win. For any questions, please contact the "International AI Challenge for Marine Debris Image Recognition" Working Group at mdivrc@digitimes.com.
