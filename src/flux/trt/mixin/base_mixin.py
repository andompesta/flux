#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import dtype

from abc import ABC, abstractmethod
from typing import Any


class BaseMixin(ABC):
    def __init__(self, should_be_dtype: dtype, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.should_be_dtype = should_be_dtype

    @abstractmethod
    def get_mixin_params(self) -> dict[str, Any]:
        pass
