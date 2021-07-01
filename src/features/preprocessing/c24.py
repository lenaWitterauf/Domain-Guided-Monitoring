import dataclass_cli
import dataclasses
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .base import Preprocessor


@dataclass_cli.add
@dataclasses.dataclass
class C24PreprocessorConfig:
    grouped_df_pkl: Path = Path("data/grouped_df11.pkl")
    event_column: str = "Event"
    fraud_column: str = "Fraud"


class C24FraudPreprocessor(Preprocessor):
    events = [
        "CREATE_TO_NOTEPAD",
        "DELETE_FROM_NOTEPAD",
        "SHOW_AD_FROM_NOTEPAD",
        "DETAIL_PAGE_VIEW",
        "DEALER_PAGE_STICKYNAV",
        "DETAIL_IMAGES_VIEW",
        "DETAIL_IMAGES_SMALL_VIEW",
        "DETAIL_PAGE_SELLER_DETAIL_VIEW",
        "DETAIL_PAGE_SELLER_STICKY_NAV",
        "DETAIL_PAGE_SHOW_SELLER_PAGE",
        "DETAIL_PAGE_GOTO_CHECKOUT_RATINGS",
        "DEALER_PAGE_SHOW_LOCATION",
        "DEALER_PAGE_TESTIMONIALS",
        "DEALER_PAGE_QUESTIONS",
        "DEALER_FAQ",
        "DEALER_REGISTRATION",
        "DEALER_OFFER_DETAIL",
        "SEARCH_MANUAL_PRICE_SORTING",
        "SEARCH_SORTING_NEWEST",
        "SEARCH_FILTER_BY_PRICE",
        "SEARCH_FILTER_BY_MODEL",
        "SEARCH_FILTER_BY_COUNTRY",
        "SEARCH_FILTER_BY_YEARBOUGHT",
        "SEARCH_FILTER_BY_YEAR",
        "SEARCH_FILTER_BY_CASE_DIAMETER",
        "SENT_CONTACT_MAIL_TO_DEALER",
        "DETAIL_PAGE_BUTTON_CONTACT_CLICKED",
        "DETAIL_PAGE_BUTTON_PURCHASE_CLICKED",
        "DETAIL_PAGE_BUTTON_PRICE_SUGGESTION_CLICKED",
        "OFFER_REQUESTED_PRICE_NEGOTIATION",
        "OFFER_REQUESTED_TRUSTED_CHECKOUT",
        "OFFER_REQUESTED_DIRECT_CHECKOUT",
        "FAQ_VIEW",
        "SEARCH_FOR_PRODUCT",
        "SEARCH_FOR_MANUFACTURER",
        "SEARCH_FOR_MODEL",
        "SEARCH_RESULT_VIEW",
        "SEARCH_TASK_CREATE",
        "TRUSTED_CHECKOUT_INFO_VIEW",
        "ABOUT_US_VIEW",
        "MAIL_VIEW",
        "HOMEPAGE_VIEW",
        "MAGAZINE_VIEW",
        "VALUATION_VIEW",
        "MYCHRONO_LANDINGPAGE",
        "USER_REGISTRATION",
        "USER_WELCOME",
        "NEWSLETTER_SUBSCRIBE",
        "WATCH_COLLECTION_CREATE",
        "WATCH_COLLECTION_VIEW_OWN_ITEM",
        "WATCH_COLLECTION_VIEW_FOLLOWED_ITEM",
        "WATCH_COLLECTION_DELETE_ITEM",
        "WATCH_COLLECTION_MOVED_FOLLOWED_ITEM",
        "PRIVATE_SELLER_AD_STARTED",
        "PRIVATE_SELLER_AD_CREATE",
        "PRIVATE_SELLER_DEALER",
        "PRIVATE_SELLER_GUIDE",
        "PRIVATE_SELLER_FAQ",
        "PRIVATE_SELLER_VIEW",
        "PRIVATE_SELLER_TESTIMONIALS",
        "PRIVATE_SELLER_QUESTIONS",
        "WILL_IT_FIT",
        "SHOW_CONTACT_NUMBERS",
        "USER_REGISTER_LANDING_PAGE",
        "AUCTION_LANDING_PAGE",
        "WATCH_NOTIFY_AVAILABLE",
        "TARGET_REGISTRATION",
        "TARGET_PRIVATE_AD_CREATE",
        "TARGET_DEALER_REGISTRATION",
        "TARGET_CHECKOUT_STARTED",
        "TARGET_CONTACTS",
        "TARGET_SEARCH_TASK",
        "TARGET_NOTEPAD_CREATE",
        "TARGET_WATCH_COLLECTION_OWNED",
        "APPS_HOME",
        "DB_AD_CREATE",
        "DB_MAIL_SENT",
        "DB_MAIL_RECEIVED",
        "DB_NOTEPAD_CREATE",
        "DB_NOTEPAD_DELETE",
        "DB_WATCH_COLLECTION_CREATE",
        "DB_UPLOAD",
        "DB_UPLOAD_REQUEST",
        "DB_SEARCH_TASK_CREATE",
        "DB_SEARCH_TASK_DELETE",
        "DB_FIRST_LOGIN",
    ]

    def __init__(
        self,
        df_pkl: Path = Path("data/grouped_df11.pkl"),
        event_column: str = "Event",
        fraud_column: str = "Fraud",
        sequence_column_name: str = "Combined",
    ):
        self.df_pkl = df_pkl
        self.event_column = event_column
        self.fraud_column = fraud_column
        self.sequence_column_name = sequence_column_name

    def load_data(self) -> pd.DataFrame:
        df = pd.read_pickle(self.df_pkl)
        df[self.fraud_column] = df[self.fraud_column].apply(
            lambda x: "fraud" if str(x) == "1" else "no_fraud"
        )
        df[self.sequence_column_name] = df[
            [self.event_column, self.fraud_column]
        ].apply(lambda x: [[self.events[int(event)], x[1]] for event in x[0]], axis=1)
        df[self.fraud_column] = df[self.fraud_column].apply(lambda x: [["fraud"]])
        df[self.event_column] = df[self.event_column].apply(
            lambda x: [[self.events[int(event)] for event in x]]
        )
        return df[[self.event_column, self.fraud_column, self.sequence_column_name]]


class C24HierarchyPreprocessor(Preprocessor):
    def load_data(self) -> pd.DataFrame:
        events = C24FraudPreprocessor.events
        hierarchy_df = pd.DataFrame(columns=["parent", "child"])
        for event in tqdm(
            events, desc="Generating hierarchy knowledge for C24 event names"
        ):
            splitted_event = event.split("_")
            for split in splitted_event:
                hierarchy_df = hierarchy_df.append(
                    {"parent": "root", "child": split,}, ignore_index=True
                )
                hierarchy_df = hierarchy_df.append(
                    {"parent": split, "child": event,}, ignore_index=True
                )

        hierarchy_df["parent_id"] = hierarchy_df["parent"]
        hierarchy_df["parent_name"] = hierarchy_df["parent"]
        hierarchy_df["child_id"] = hierarchy_df["child"]
        hierarchy_df["child_name"] = hierarchy_df["child"]
        return hierarchy_df[["parent_id", "child_id", "parent_name", "child_name"]]

