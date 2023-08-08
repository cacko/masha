from datetime import datetime
import logging
from typing import Any, Optional
from masha.firebase.service_account import db

class OptionsDb(object):

    @property
    def root_ref(self):
        return db.reference(f"app/")

    def options(self, **kwds):
        options_ref = self.root_ref.child("options")
        return options_ref.set(kwds)


class AccessDb(object):

    @property
    def root_ref(self):
        return db.reference(f"app/")

    def access(self, **kwds):
        options_ref = self.root_ref.child("access")
        return options_ref.set(kwds)
